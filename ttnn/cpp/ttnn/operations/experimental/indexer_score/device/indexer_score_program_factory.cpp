// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "indexer_score_program_factory.hpp"

#include <algorithm>
#include <array>
#include <utility>
#include <vector>

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/work_split.hpp>
#include "hostdevcommon/kernel_structs.h"  // tt::CBIndex

#include "ttnn/operations/transformer/sdpa/device/sdpa_subblock_utils.hpp"
#include "kernels/indexer_score_cb.hpp"          // shared host/device CB-index argument layout (CbArg)
#include "kernels/indexer_score_work_split.hpp"  // shared host/device causal work-split formula

namespace ttnn::operations::experimental::indexer_score::program {

// Runtime-arg slots, shared by create()/override_runtime_arguments() and matched positionally
// by the kernels. Reader: q,k,w addrs; writer: out addr.
namespace rt_arg {
constexpr uint32_t reader_q_addr = 0;
constexpr uint32_t reader_k_addr = 1;
constexpr uint32_t reader_w_addr = 2;
constexpr uint32_t writer_out_addr = 0;
}  // namespace rt_arg

// Patch one runtime-arg slot on a program-cache hit, asserting the slot exists.
inline void patch_arg(tt::tt_metal::RuntimeArgsData& args, uint32_t index, uint32_t value, const char* name) {
    TT_FATAL(index < args.size(), "indexer_score override: {} index {} >= args size {}", name, index, args.size());
    args[index] = value;
}

// Dense deal landed exactly on the grid (q/w mcast along rows, k down columns) and each direction's
// lines form contiguous NoC rects. 0 = direction off (per-core DRAM read); see call-site block.
struct McastPlan {
    bool grid_aligned = false;
    uint32_t k_mcast_on = 0;  // K columns are vertical NoC rects
    uint32_t q_mcast_on = 0;  // Q/W rows are single horizontal NoC rects (covers q and w)
};

// Pure analysis of the physical core coords: grid alignment + per-line NoC-rect contiguity for the
// two mcast directions. phys is indexed row-major (y * grid.x + x).
inline McastPlan compute_mcast_plan(
    CoreCoord grid,
    const std::vector<CoreCoord>& phys,
    uint32_t groups,
    uint32_t num_cores,
    uint32_t base,
    uint32_t rem,
    uint64_t total_units,
    uint32_t HB,
    uint32_t Hi) {
    const auto cidx = [&](uint32_t x, uint32_t y) { return y * grid.x + x; };
    // grid-aligned iff group g == grid row y and the row's grid_x cores evenly split its k-chunks
    // (units_per_group == grid.x * base, no remainder, full grid).
    const uint32_t units_per_group = groups > 0 ? (uint32_t)(total_units / groups) : 0;
    const bool grid_aligned =
        groups == grid.y && num_cores == (uint32_t)(grid.x * grid.y) && rem == 0 && units_per_group == grid.x * base;

    // K columns must be vertical NoC rects: shared x down the column, contiguous y spanning the grid.
    bool k_cols_ok = grid_aligned;
    for (uint32_t x = 0; x < grid.x && k_cols_ok; ++x) {
        const uint32_t px = phys[cidx(x, 0)].x;
        uint32_t ymin = phys[cidx(x, 0)].y, ymax = ymin;
        for (uint32_t y = 0; y < grid.y; ++y) {
            const auto& p = phys[cidx(x, y)];
            if (p.x != px) {
                k_cols_ok = false;
            }
            ymin = std::min<uint32_t>(ymin, p.y);
            ymax = std::max<uint32_t>(ymax, p.y);
        }
        if (ymax - ymin + 1 != grid.y) {
            k_cols_ok = false;
        }
    }
    // Q/W row-mcast needs every core in a logical row on ONE physical NoC row (mcast = the row's
    // horizontal bounding-box rect) and all heads resident (one q block). x-contiguity not required
    // (the NoC routes the bbox rect); only a row spanning multiple physical NoC rows disables it.
    // grid.x >= grid.y guards the diagonal sender lookup phys[cidx(y, y)] at the call site: cidx(y, y)
    // is in-bounds only while y < grid.x, which holds for every row iff the grid is at least as wide as
    // tall (always true on Blackhole's 14x10 grid; the guard keeps a narrower grid safe).
    bool q_rows_ok = grid_aligned && HB == Hi && grid.x >= grid.y;
    for (uint32_t y = 0; y < grid.y && q_rows_ok; ++y) {
        const uint32_t py = phys[cidx(0, y)].y;
        for (uint32_t x = 0; x < grid.x; ++x) {
            if (phys[cidx(x, y)].y != py) {
                q_rows_ok = false;
            }
        }
    }
    return {grid_aligned, k_cols_ok ? 1u : 0u, q_rows_ok ? 1u : 0u};
}

// Output-stationary flat deal of causal-valid work units. One unit = QC q-tile-rows x up-to-KC
// k-tiles, dealt evenly across cores row-major (kernels invert the flat index). Heads stream in
// HB-head groups; fully-future tiles get the full -inf mask, row tails -inf-filled by the writer
// (zeros unsafe: gates can be negative).
IndexerScoreProgramFactory::cached_program_t IndexerScoreProgramFactory::create(
    const operation_attributes_t& args, const tensor_args_t& tensors, tensor_return_value_t& out) {
    tt::tt_metal::Program program = tt::tt_metal::CreateProgram();

    const auto& q = tensors.q;
    const auto& k = tensors.k;
    const auto& w = tensors.weights;

    // Inputs and knobs are validated in IndexerScoreDeviceOperation::validate_on_program_cache_miss;
    // here we only derive tile dims and the one build-specific (subblock) constraint.
    const uint32_t Hi = q.logical_shape()[1];
    const uint32_t Sq = q.logical_shape()[2];
    const uint32_t D = q.logical_shape()[3];
    const uint32_t T = k.logical_shape()[2];

    const uint32_t Sqt = Sq / tt::constants::TILE_HEIGHT;
    const uint32_t Tt = T / tt::constants::TILE_WIDTH;
    const uint32_t Dt = D / tt::constants::TILE_WIDTH;
    const uint32_t chunk_t = args.chunk_start_idx / tt::constants::TILE_WIDTH;

    // Work-unit knobs, converted from elements to tiles / heads.
    const auto& cfg = args.program_config;
    const uint32_t QC = cfg.q_chunk_size / tt::constants::TILE_HEIGHT;
    const uint32_t KC = cfg.k_chunk_size / tt::constants::TILE_WIDTH;
    const uint32_t HB = resolve_head_group(cfg, Hi);

    // Compute knobs from the resolved compute config. math_fidelity defaults to the dtype-derived
    // choice (bf16 -> HiFi2, both bfp8 -> LoFi) but a caller can override it; validate guarantees
    // fp32_dest_acc_en==false / dst_full_sync_en==false (the bf16-DEST half-sync layout the custom
    // blocked bcast-col MUL LLK + packer path are validated for).
    const auto [math_fidelity, math_approx_mode, fp32_dest_acc_en, packer_l1_acc, dst_full_sync_en] =
        ttnn::get_compute_kernel_config_args(q.device()->arch(), args.compute_kernel_config);

    // qk matmul subblock: heads are output rows, k column is 1 tile wide (SDPA-style), so only the
    // subblock height (head rows) is needed.
    const uint32_t dst_size = fp32_dest_acc_en ? 4 : 8;  // half-sync, as in sdpa_program_factory
    const uint32_t qk_subblock_h = ttnn::prim::detail::determine_largest_subblock_size(HB, 1, dst_size).first;
    TT_FATAL(HB % qk_subblock_h == 0, "head group {} must be divisible by qk_subblock_h={}", HB, qk_subblock_h);

    // QC/KC/HB are verbatim from the config -- no auto-tune; the caller owns the perf trade-off (see
    // glx_config() in the test). An oversized config is not clamped; it fails at CB allocation.

    // ---- generalized banded product schedule -------------------------------------------------
    // Work space = G groups (q-row-groups) x U k-bands. Map groups -> grid ROWS (q/w shared along a row
    // => q-mcast) and bands -> grid COLUMNS (k shared down a column => k-mcast). Rows phase-stack groups
    // when G > grid.y; each column owns a contiguous even band chunk so a non-divisible U still tiles and
    // every core in a column reads an identical band chunk (k-mcast lockstep). Always uses the full
    // rows_used x cols_used rectangle of the grid.
    const uint32_t G = Sqt / QC;                // q-row-groups
    const uint32_t U = units_in_group(KC, Tt);  // k-bands per group (ceil(Tt/KC))
    const auto grid = q.device()->compute_with_storage_grid_size();
    const uint32_t gx = grid.x, gy = grid.y;

    const uint32_t rows_used = rows_used_for(G, gy);  // grid rows carrying groups (divides G when G>gy)
    const uint32_t cols_used = cols_used_for(U, gx);  // grid cols carrying band-chunks
    const uint32_t num_cores = rows_used * cols_used;

    // Contiguous even band split across the used columns: first (U % cols_used) cols get one extra band.
    std::vector<uint32_t> col_band_start(cols_used), col_band_size(cols_used);
    {
        const uint32_t bpc = U / cols_used, extra = U % cols_used;
        uint32_t off = 0;
        for (uint32_t x = 0; x < cols_used; ++x) {
            col_band_size[x] = bpc + (x < extra ? 1u : 0u);
            col_band_start[x] = off;
            off += col_band_size[x];
        }
    }
    // Groups per row: rows_used divides G, so every row runs the SAME num_groups = G/rows_used groups
    // (group = y + p*rows_used, p in [0,num_groups)). Uniform across all rows => every column takes the
    // same number of group-phases => k-mcast down the column stays lockstep (no sender/receiver hang).
    const uint32_t num_groups = G / rows_used;

    const CoreRange core_rect(CoreCoord{0, 0}, CoreCoord{cols_used - 1, rows_used - 1});
    const CoreRangeSet core_ranges(core_rect);

    // Physical coords of the used rectangle, indexed [y][x], for the mcast bounding boxes below.
    std::vector<std::vector<CoreCoord>> phys2(rows_used, std::vector<CoreCoord>(cols_used));
    for (uint32_t y = 0; y < rows_used; ++y) {
        for (uint32_t x = 0; x < cols_used; ++x) {
            phys2[y][x] = q.device()->worker_core_from_logical_core(CoreCoord{x, y});
        }
    }

    // k-mcast down a column needs >1 row; q-mcast along a row needs >1 col AND resident heads (HB==Hi):
    // streaming reads q per output tile, a pattern the row mcast doesn't cover (k-mcast is HB-independent).
    const uint32_t k_mcast_on = (rows_used > 1) ? 1u : 0u;
    const uint32_t q_mcast_on = (cols_used > 1 && HB == Hi) ? 1u : 0u;

    // 3 semaphores per active direction: send (receivers ready), recv (sender relays valid in), valid
    // (constant 1, relay source). Mirrors SDPA chain_link's handshake.
    const uint32_t k_send_sem = k_mcast_on ? tt::tt_metal::CreateSemaphore(program, core_ranges, 0) : 0;
    const uint32_t k_recv_sem = k_mcast_on ? tt::tt_metal::CreateSemaphore(program, core_ranges, 0) : 0;
    const uint32_t k_valid_sem = k_mcast_on ? tt::tt_metal::CreateSemaphore(program, core_ranges, 1) : 0;
    const uint32_t q_send_sem = q_mcast_on ? tt::tt_metal::CreateSemaphore(program, core_ranges, 0) : 0;
    const uint32_t q_recv_sem = q_mcast_on ? tt::tt_metal::CreateSemaphore(program, core_ranges, 0) : 0;
    const uint32_t q_valid_sem = q_mcast_on ? tt::tt_metal::CreateSemaphore(program, core_ranges, 1) : 0;

    const uint32_t bf16_tile = tt::tile_size(tt::DataFormat::Float16_b);
    const uint32_t fp32_tile = tt::tile_size(tt::DataFormat::Float32);

    // q (srcB) and k (srcA) are matmul inputs; either may be bfp8_b to halve its DRAM/L1 footprint
    // (w stays bf16). The format flows into the CB; the compute kernel needs no change.
    const bool q_is_bfp8 = q.dtype() == tt::tt_metal::DataType::BFLOAT8_B;
    const bool k_is_bfp8 = k.dtype() == tt::tt_metal::DataType::BFLOAT8_B;
    const tt::DataFormat q_fmt = q_is_bfp8 ? tt::DataFormat::Bfp8_b : tt::DataFormat::Float16_b;
    const tt::DataFormat k_fmt = k_is_bfp8 ? tt::DataFormat::Bfp8_b : tt::DataFormat::Float16_b;
    const uint32_t q_tile = tt::tile_size(q_fmt);
    const uint32_t k_tile = tt::tile_size(k_fmt);

    // Continuous CB indices, allocated on demand: each make_cb claims the next free index (c_0, c_1, ...)
    // and records it under its CbArg slot. The whole array is forwarded to the kernels as compile-time
    // args (CbArg is the shared slot order; see indexer_score_cb.hpp), so every buffer is resolved through
    // this one allocation and indices can never drift host<->device.
    std::array<uint32_t, num_cb_args> cb_id{};
    uint32_t next_cb_index = tt::CBIndex::c_0;
    auto make_cb = [&](uint32_t slot, uint32_t ntiles, tt::DataFormat fmt, uint32_t tile_bytes) {
        const uint32_t idx = next_cb_index++;
        cb_id[slot] = idx;
        tt::tt_metal::CreateCircularBuffer(
            program,
            core_ranges,
            tt::tt_metal::CircularBufferConfig(ntiles * tile_bytes, {{idx, fmt}}).set_page_size(idx, tile_bytes));
    };

    const bool stream_heads = HB < Hi;

    // Allocate each CB by its CbArg slot; make_cb assigns the next continuous index.
    make_cb(cb_q_arg, (stream_heads ? 2 : 1) * HB * QC * Dt, q_fmt, q_tile);
    make_cb(cb_k_arg, 2 * KC * Dt, k_fmt, k_tile);
    make_cb(cb_w_arg, Hi * QC, tt::DataFormat::Float16_b, bf16_tile);
    make_cb(cb_mask_arg, num_mask_tiles, tt::DataFormat::Float16_b, bf16_tile);
    const tt::DataFormat acc_fmt = fp32_dest_acc_en ? tt::DataFormat::Float32 : tt::DataFormat::Float16_b;
    const uint32_t acc_tile = fp32_dest_acc_en ? fp32_tile : bf16_tile;
    // cb_qk buffers a batch of the group's relu(q.kT) tiles so compute runs that batch's matmuls then
    // its mul+accumulates, hoisting the matmul<->eltwise reinit out of the per-head-pass loop.
    // QC==1 has spare L1 -> batch the whole group; QC>1 doubles cb_q/cb_w, so cap at 32.
    const uint32_t qk_batch_cap = (QC == 1) ? HB : 32u;
    const uint32_t qk_batch_heads = std::min<uint32_t>(HB, qk_batch_cap);  // multiple of qk_subblock_h
    // The compute kernel walks HB in qk_batch_heads-sized chunks (chunk += qk_batch_heads), so HB must
    // be a whole multiple or the last chunk over-reads past the resident head group. Only reachable when
    // the 32-cap engages (HB > 32 && QC > 1); the deployed cases never hit it, but guard loudly.
    TT_FATAL(
        HB % qk_batch_heads == 0,
        "head_group {} not divisible by qk_batch_heads {} (QC>1 with HB>32); reduce head_group_size or q_chunk_size",
        HB,
        qk_batch_heads);
    // Full-strip path batches the whole k chunk's columns per matmul<->mul mode switch (one switch per
    // batch, not per output tile): w is column-independent and the whole k chunk is resident, so a row's
    // columns share one switch. Only when the group is a single chunk (qk_batch_heads == Hi) and the fast
    // strip is used (KC >= 2); cb_qk then holds KC * qk_batch_heads tiles and fails at allocation if the
    // caller's k_chunk_size is too large for L1 (the caller owns the knob trade-off).
    const bool single_chunk = (qk_batch_heads == Hi) && !stream_heads;
    const uint32_t qk_col_batch = (KC >= 2 && single_chunk) ? KC : 1u;
    make_cb(cb_qk_arg, qk_col_batch * qk_batch_heads, acc_fmt, acc_tile);
    // cb_out_strip holds a unit's untilized output. Uniform KC push/pop keeps the packer's KC-tile
    // reads contiguous (a non-uniform push would wrap the ring mid-strip).
    make_cb(cb_out_strip_arg, 2 * KC, tt::DataFormat::Float16_b, bf16_tile);
    // cb_acc_strip accumulates a whole unit's QC*KC strip, then all QC strips untilize under ONE
    // pack_untilize bracket (per-strip cost amortizes over QC*KC, not KC). max(2*KC, .) keeps the
    // QC<=2 double buffer and a whole multiple of the QC*KC batch so a uniform push never wraps mid-unit.
    make_cb(cb_acc_strip_arg, std::max(2u * KC, QC * KC), acc_fmt, acc_tile);

    // No up-front L1-fit guard: an oversized QC/KC/head_group config fails at CB allocation. The caller
    // owns the knob trade-off (see glx_config() in the test).

    // Common args: 8 dims then the CB indices in CbArg order (kernels read both from this shared base).
    std::vector<uint32_t> common_ct = {Hi, Sqt, Tt, Dt, chunk_t, QC, KC, HB};
    common_ct.insert(common_ct.end(), cb_id.begin(), cb_id.end());

    std::vector<uint32_t> reader_ct = common_ct;
    tt::tt_metal::TensorAccessorArgs(*q.buffer()).append_to(reader_ct);
    tt::tt_metal::TensorAccessorArgs(*k.buffer()).append_to(reader_ct);
    tt::tt_metal::TensorAccessorArgs(*w.buffer()).append_to(reader_ct);
    // multicast: on/off per direction (q_mcast_on covers q and w) then the 6 semaphore ids, at fixed
    // offsets right after the three TensorAccessors.
    reader_ct.push_back(k_mcast_on);
    reader_ct.push_back(q_mcast_on);
    reader_ct.push_back(k_send_sem);
    reader_ct.push_back(k_recv_sem);
    reader_ct.push_back(k_valid_sem);
    reader_ct.push_back(q_send_sem);
    reader_ct.push_back(q_recv_sem);
    reader_ct.push_back(q_valid_sem);

    std::vector<uint32_t> writer_ct = common_ct;
    const uint32_t out_elem_bytes = out.element_size();  // from the output tensor's dtype (bf16 today)
    writer_ct.push_back(T * out_elem_bytes);             // row-major page = one full row of T scores
    tt::tt_metal::TensorAccessorArgs(*out.buffer()).append_to(writer_ct);

    std::vector<uint32_t> compute_ct = common_ct;
    compute_ct.push_back(qk_subblock_h);
    compute_ct.push_back(qk_batch_heads);  // head tiles per matmul/mul phase chunk
    compute_ct.push_back(qk_col_batch);    // k-columns batched per mode switch in the full-strip path

    const std::string kdir = "ttnn/cpp/ttnn/operations/experimental/indexer_score/device/kernels/";
    auto reader_id = tt::tt_metal::CreateKernel(
        program, kdir + "reader_indexer_score.cpp", core_ranges, tt::tt_metal::ReaderDataMovementConfig(reader_ct));
    auto writer_id = tt::tt_metal::CreateKernel(
        program, kdir + "writer_indexer_score.cpp", core_ranges, tt::tt_metal::WriterDataMovementConfig(writer_ct));
    auto compute_id = tt::tt_metal::CreateKernel(
        program,
        kdir + "compute_indexer_score.cpp",
        core_ranges,
        tt::tt_metal::ComputeConfig{
            .math_fidelity = math_fidelity,
            .fp32_dest_acc_en = fp32_dest_acc_en,
            .dst_full_sync_en = dst_full_sync_en,
            .math_approx_mode = math_approx_mode,
            .compile_args = compute_ct});

    // Per-core args: schedule {row_group0, group_stride, num_groups, band0, num_bands} then (reader only)
    // the K-column + Q/W-row mcast 8-tuples. role is a McastRole (none = per-core DRAM read); rect
    // (xs,ys,xe,ye) + sender (sx,sy) are physical NoC; ndst = #receivers. mcast rects are fixed per core
    // (one column for k, one row for q); only the data changes per group/band phase.
    const auto u32 = [](auto v) { return static_cast<uint32_t>(v); };
    std::vector<CoreCoord> cores;
    cores.reserve(num_cores);
    for (uint32_t y = 0; y < rows_used; ++y) {
        // physical bbox of row y across the used columns (q/w mcast rect); py constant along the row.
        uint32_t q_xs = u32(phys2[y][0].x), q_xe = u32(phys2[y][0].x);
        for (uint32_t x = 0; x < cols_used; ++x) {
            q_xs = std::min<uint32_t>(q_xs, u32(phys2[y][x].x));
            q_xe = std::max<uint32_t>(q_xe, u32(phys2[y][x].x));
        }
        const uint32_t q_py = u32(phys2[y][0].y);
        const uint32_t q_diag = std::min<uint32_t>(y, cols_used - 1);  // diagonal sender column
        const CoreCoord q_sender = phys2[y][q_diag];
        for (uint32_t x = 0; x < cols_used; ++x) {
            // physical bbox of column x down the used rows (k mcast rect); px constant down the column.
            uint32_t k_ys = u32(phys2[0][x].y), k_ye = u32(phys2[0][x].y);
            for (uint32_t yy = 0; yy < rows_used; ++yy) {
                k_ys = std::min<uint32_t>(k_ys, u32(phys2[yy][x].y));
                k_ye = std::max<uint32_t>(k_ye, u32(phys2[yy][x].y));
            }
            const uint32_t k_px = u32(phys2[0][x].x);
            const CoreCoord k_sender = phys2[0][x];

            const CoreCoord core{x, y};
            cores.push_back(core);
            const std::array<uint32_t, 5> sched = {y, rows_used, num_groups, col_band_start[x], col_band_size[x]};

            std::vector<uint32_t> reader_rt = {q.buffer()->address(), k.buffer()->address(), w.buffer()->address()};
            reader_rt.insert(reader_rt.end(), sched.begin(), sched.end());
            const auto push_mcast_dir = [&](uint32_t role,
                                            uint32_t xs,
                                            uint32_t ys,
                                            uint32_t xe,
                                            uint32_t ye,
                                            const CoreCoord& s,
                                            uint32_t ndst) {
                reader_rt.push_back(role);
                reader_rt.push_back(xs);
                reader_rt.push_back(ys);
                reader_rt.push_back(xe);
                reader_rt.push_back(ye);
                reader_rt.push_back(u32(s.x));
                reader_rt.push_back(u32(s.y));
                reader_rt.push_back(ndst);
            };
            // K column x: sender row 0, receivers rows [1, rows_used); vertical rect spanning the column.
            push_mcast_dir(
                k_mcast_on ? (y == 0 ? mcast_role_sender : mcast_role_receiver) : mcast_role_none,
                k_px,
                k_ys,
                k_px,
                k_ye,
                k_sender,
                rows_used - 1);
            // Q/W row y: sender on the diagonal column, receivers the rest of the row; horizontal rect.
            push_mcast_dir(
                q_mcast_on ? (x == q_diag ? mcast_role_sender : mcast_role_receiver) : mcast_role_none,
                q_xs,
                q_py,
                q_xe,
                q_py,
                q_sender,
                cols_used - 1);
            tt::tt_metal::SetRuntimeArgs(program, reader_id, core, reader_rt);
            tt::tt_metal::SetRuntimeArgs(program, compute_id, core, std::vector<uint32_t>(sched.begin(), sched.end()));
            std::vector<uint32_t> writer_rt = {out.buffer()->address()};
            writer_rt.insert(writer_rt.end(), sched.begin(), sched.end());
            tt::tt_metal::SetRuntimeArgs(program, writer_id, core, writer_rt);
        }
    }

    return {
        std::move(program),
        IndexerScoreSharedVariables{
            .reader_kernel = reader_id,
            .compute_kernel = compute_id,
            .writer_kernel = writer_id,
            .worker_cores = cores}};
}

void IndexerScoreProgramFactory::override_runtime_arguments(
    cached_program_t& cached, const operation_attributes_t&, const tensor_args_t& tensors, tensor_return_value_t& out) {
    auto& shared = cached.shared_variables;
    auto& reader_args = tt::tt_metal::GetRuntimeArgs(cached.program, shared.reader_kernel);
    auto& writer_args = tt::tt_metal::GetRuntimeArgs(cached.program, shared.writer_kernel);
    for (const auto& core : shared.worker_cores) {
        auto& reader_rt = reader_args[core.x][core.y];
        patch_arg(reader_rt, rt_arg::reader_q_addr, tensors.q.buffer()->address(), "reader.q_addr");
        patch_arg(reader_rt, rt_arg::reader_k_addr, tensors.k.buffer()->address(), "reader.k_addr");
        patch_arg(reader_rt, rt_arg::reader_w_addr, tensors.weights.buffer()->address(), "reader.w_addr");
        patch_arg(writer_args[core.x][core.y], rt_arg::writer_out_addr, out.buffer()->address(), "writer.out_addr");
    }
}

}  // namespace ttnn::operations::experimental::indexer_score::program

// TF_DS_KV_SHARD kernels (dsv41_shard.cu: fetch, select and the union) run on the host, every block and thread in turn, against a plain reference.
// No nvcc or GPU: clang++ -std=c++20 -I tools/dsv41_shard_emulate tools/dsv41_shard_emulate/emulate.cpp -o /tmp/e && /tmp/e
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <random>
#define TF_DS_EMULATE 1 // the ring kernels' protocol needs a GPU; their row packing runs here
#include "../../zig/kernels/cuda/dsv41_shard.cu"
static void launch_fetch(int rows, long long kos, const uint8_t* c, const uint8_t* s, long long cb, long long sb, const long long* idx, long long ks, int kk, const long long* sbase, const long long* own, int copy, uint8_t* dc, uint8_t* ds, long long dk, long long* out, int how) {
  gridDim = {(unsigned)rows, (unsigned)((kos + 15) / 16), 1}; blockDim = {256, 1, 1};
  for (unsigned bx = 0; bx < gridDim.x; bx++) for (unsigned by = 0; by < gridDim.y; by++) for (unsigned t = 0; t < 256; t++) {
    blockIdx = {bx, by, 0}; threadIdx = {t, 0, 0};
    tf_ds_kv_fetch_kernel(c, s, cb, sb, idx, ks, kk, sbase, own, copy, dc, ds, dk, out, kos, how);
  }
}
int main() {
  std::mt19937_64 g(1);
  int fails = 0;
  for (int trial = 0; trial < 200; trial++) {
    const long long cb = 256, sb = 32;
    int rows = 1 + g() % 16; int kk = 1 + g() % 600; bool pad = g() % 2; long long kos = pad ? (kk + 15) / 16 * 16 : kk;
    int how = g() % 3; int copy = how == 2 ? 0 : (int)(g() % 2); long long dk = kk + (how == 0 ? g() % 4 : 0);
    long long plane_rows = 5000;
    std::vector<uint8_t> c(plane_rows * cb), s(plane_rows * sb);
    for (auto& x : c) x = g(); for (auto& x : s) x = g();
    std::vector<long long> idx(rows * kk), sbase(rows), own(rows);
    for (int r = 0; r < rows; r++) { sbase[r] = g() % 1000; own[r] = g() % 2; for (int j = 0; j < kk; j++) idx[r * kk + j] = (g() % 7 == 0) ? -1 : (long long)(g() % 3000); }
    bool use_s = g() % 2, use_o = g() % 2;
    std::vector<uint8_t> dc(rows * dk * cb, 0xAB), ds(rows * dk * sb, 0xCD), ec = dc, es = ds;
    std::vector<long long> out(rows * kos, 77), eo(rows * kos);
    launch_fetch(rows, kos, c.data(), s.data(), cb, sb, idx.data(), kk, kk, use_s ? sbase.data() : nullptr, use_o ? own.data() : nullptr, copy, dc.data(), ds.data(), dk, out.data(), how);
    for (int r = 0; r < rows; r++) for (long long j = 0; j < kos; j++) {
      long long v = j < kk ? idx[r * kk + j] : -1; bool mine = !use_o || own[r] != 0; long long b = use_s ? sbase[r] : 0;
      eo[r * kos + j] = how == 2 ? (mine ? v : -1) : (v < 0 ? -1 : (how == 1 ? r * dk + j : j));
      if (copy && how != 2 && v >= 0 && mine) { memcpy(&ec[(r * dk + j) * cb], &c[(b + v) * cb], cb); memcpy(&es[(r * dk + j) * sb], &s[(b + v) * sb], sb); }
    }
    if (out != eo || dc != ec || ds != es) { fails++; printf("fetch trial %d differs (how %d copy %d)\n", trial, how, copy); }
  }
  for (int trial = 0; trial < 50; trial++) {
    int rows = 1 + g() % 16, planes = 1 + g() % 4; long long bytes = 16 * (1 + g() % 600), pb = rows * bytes;
    std::vector<uint8_t> src(planes * pb), dst(planes * pb); for (auto& x : src) x = g(); for (auto& x : dst) x = g();
    std::vector<long long> sel(rows); for (auto& x : sel) x = g() % 2;
    auto exp = dst; for (int p = 0; p < planes; p++) for (int r = 0; r < rows; r++) if (sel[r] == 0) memcpy(&exp[p * pb + r * bytes], &src[p * pb + r * bytes], bytes);
    gridDim = {(unsigned)rows, (unsigned)planes, 1}; blockDim = {256, 1, 1};
    for (unsigned bx = 0; bx < gridDim.x; bx++) for (unsigned by = 0; by < gridDim.y; by++) for (unsigned t = 0; t < 256; t++) {
      blockIdx = {bx, by, 0}; threadIdx = {t, 0, 0};
      tf_ds_select_rows_kernel((const uint4*)src.data(), (uint4*)dst.data(), sel.data(), 0, bytes / 16, pb / 16);
    }
    if (dst != exp) { fails++; printf("select trial %d differs\n", trial); }
  }
  // the union: mark, scan (three passes), compact, renumber against a plain reference
  auto run = [](unsigned gx, unsigned gy, unsigned bx_, auto&& k) {
    gridDim = {gx, gy, 1}; blockDim = {bx_, 1, 1};
    for (unsigned bx = 0; bx < gx; bx++) for (unsigned by = 0; by < gy; by++) for (unsigned t = 0; t < bx_; t++) { blockIdx = {bx, by, 0}; threadIdx = {t, 0, 0}; k(); }
  };
  for (int trial = 0; trial < 60; trial++) {
    const long long cb = 256, sb = 32;
    int rows = 1 + g() % 40; int kk = 1 + g() % 600; long long kos = (g() % 2) ? (kk + 15) / 16 * 16 : kk;
    long long limit = trial == 0 ? 1 : 1 + g() % (trial % 3 == 0 ? 5000 : 200000);
    std::vector<long long> idx(rows * kk);
    for (auto& v : idx) { long long x = g() % (limit + limit / 8 + 2); v = (g() % 9 == 0) ? -1 : x; } // some past limit
    std::vector<int> mark(limit, 0), sums(TF_DS_SCAN_SEGS + 1, 77);
    run(rows, (kk + 255) / 256, 256, [&] { tf_ds_union_mark_kernel(idx.data(), kk, kk, mark.data(), limit); });
    run(TF_DS_SCAN_SEGS / 256, 1, 256, [&] { tf_ds_union_sums_kernel(mark.data(), limit, sums.data()); });
    run(1, 1, 1, [&] { tf_ds_union_offsets_kernel(sums.data()); });
    run(TF_DS_SCAN_SEGS / 256, 1, 256, [&] { tf_ds_union_scan_kernel(mark.data(), limit, sums.data()); });
    std::vector<int> ref(limit, 0); for (auto v : idx) if (v >= 0 && v < limit) ref[v] = 1;
    int c = 0; for (auto& x : ref) { c += x; x = c; }
    if (mark != ref || sums[TF_DS_SCAN_SEGS] != c) { fails++; printf("union scan trial %d differs\n", trial); continue; }
    std::vector<uint8_t> pc(limit * cb), ps(limit * sb); for (auto& x : pc) x = g(); for (auto& x : ps) x = g();
    std::vector<uint8_t> dc(c * cb + 16, 0xAB), ds(c * sb + 16, 0xCD), ec = dc, es = ds;
    run((unsigned)((limit + 15) / 16), 1, 256, [&] { tf_ds_union_compact_kernel(pc.data(), ps.data(), cb, sb, mark.data(), limit, dc.data(), ds.data()); });
    long long d = 0; for (long long t = 0; t < limit; t++) if (ref[t] != (t ? ref[t - 1] : 0)) { memcpy(&ec[d * cb], &pc[t * cb], cb); memcpy(&es[d * sb], &ps[t * sb], sb); d++; }
    std::vector<long long> out(rows * kos, 77), eo(rows * kos);
    run(rows, (unsigned)((kos + 255) / 256), 256, [&] { tf_ds_union_renumber_kernel(idx.data(), kk, kk, mark.data(), limit, out.data(), kos); });
    for (int r = 0; r < rows; r++) for (long long j = 0; j < kos; j++) { long long v = j < kk ? idx[r * kk + j] : -1; eo[r * kos + j] = v >= 0 && v < limit ? ref[v] - 1 : -1; }
    if (dc != ec || ds != es || out != eo) { fails++; printf("union trial %d differs\n", trial); continue; }
    // what the attention reads: the renumbered picks over the copy are the original picks over the plane
    for (int r = 0; r < rows; r++) for (int j = 0; j < kk; j++) { long long v = idx[r * kk + j]; if (v < 0 || v >= limit) continue; long long o = out[r * kos + j];
      if (memcmp(&dc[o * cb], &pc[v * cb], cb) || memcmp(&ds[o * sb], &ps[v * sb], sb)) { fails++; printf("union reads trial %d differ\n", trial); r = rows; break; } }
  }
  // the ring exchanges' row packing (tf_ds_ring_pack / unpack, the stage and collect kernels' copies): any grid's
  // threads together give shard_ring.zig's stageRef / collectRef slot and rows
  for (int trial = 0; trial < 200; trial++) {
    TfDsRingRows s{}; int nseg = 1 + g() % TF_DS_RING_SEGS; int rows = 1 + g() % 48;
    // a host-made list (shard_ring.lists: the count, then rows in slot order; any order the host gives is kept)
    int list[TF_DS_RING_ROWS + 1] = {0}; int n = 0;
    for (int r = 0; r < rows; r++) if (g() % 3 == 0) list[1 + n++] = r;
    if (g() % 4 == 0) for (int i = n - 1; i > 0; i--) { int j = g() % (i + 1); int t = list[1 + i]; list[1 + i] = list[1 + j]; list[1 + j] = t; }
    list[0] = n;
    std::vector<std::vector<uint4>> src(nseg), dst(nseg), exp(nseg);
    long long row_total = 0;
    for (int k = 0; k < nseg; k++) {
      s.row16[k] = 1 + g() % 40; s.src16[k] = s.row16[k] + g() % 3; s.dst16[k] = s.row16[k] + g() % 3; row_total += s.row16[k];
      src[k].resize(rows * s.src16[k]); dst[k].resize(rows * s.dst16[k]);
      for (auto& w : src[k]) w = {(unsigned)g(), (unsigned)g(), (unsigned)g(), (unsigned)g()};
      for (auto& w : dst[k]) w = {(unsigned)g(), (unsigned)g(), (unsigned)g(), (unsigned)g()};
      exp[k] = dst[k]; s.src[k] = src[k].data(); s.dst[k] = dst[k].data();
    }
    const int* lr = list + 1;
    if (tf_ds_ring_count(list) != n || tf_ds_ring_words(s, nseg, n) != n * row_total) { fails++; printf("ring count trial %d differs\n", trial); continue; }
    std::vector<uint4> slot(rows * row_total + 1), ref(rows * row_total + 1);
    long long off = 0;
    for (int k = 0; k < nseg; k++) { for (int c = 0; c < n; c++) for (int w = 0; w < s.row16[k]; w++) ref[off + c * s.row16[k] + w] = src[k][lr[c] * s.src16[k] + w]; off += (long long)n * s.row16[k]; }
    const unsigned T = 1 + g() % 3000; // a grid's threads
    for (unsigned t = 0; t < T; t++) tf_ds_ring_pack(s, nseg, lr, n, slot.data(), t, T);
    if (memcmp(slot.data(), ref.data(), n * row_total * 16)) { fails++; printf("ring pack trial %d differs\n", trial); continue; }
    off = 0;
    for (int k = 0; k < nseg; k++) { for (int c = 0; c < n; c++) for (int w = 0; w < s.row16[k]; w++) exp[k][lr[c] * s.dst16[k] + w] = ref[off + c * s.row16[k] + w]; off += (long long)n * s.row16[k]; }
    const unsigned U = 1 + g() % 3000;
    for (unsigned t = 0; t < U; t++) tf_ds_ring_unpack(s, nseg, lr, n, ref.data(), t, U);
    for (int k = 0; k < nseg; k++) if (memcmp(dst[k].data(), exp[k].data(), dst[k].size() * 16)) { fails++; printf("ring unpack trial %d seg %d differs\n", trial, k); break; }
  }
  // a list count out of range is clamped (never indexes past the list)
  { int bad[TF_DS_RING_ROWS + 1] = {0}; bad[0] = -3; int a = tf_ds_ring_count(bad); bad[0] = 1000; int b = tf_ds_ring_count(bad);
    if (a != 0 || b != TF_DS_RING_ROWS) { fails++; printf("ring count clamp differs\n"); } }
  printf("%s (%d failures)\n", fails ? "FAIL" : "PASS", fails);
  return fails != 0;
}

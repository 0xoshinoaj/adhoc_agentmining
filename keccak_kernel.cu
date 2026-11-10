// AMNT Plan A 解答搜尋 kernel：keccak-f[1600]，只計算，不含任何金鑰或簽名。
// 輸入 pre[25]：已吸收第 1 個 136-byte 區塊並 XOR 好第 2 區塊固定部分的 Keccak 狀態。
// 每個 thread 只把 nonce 低 64 bit（lane 6）XOR 進去、做一次 keccak-f，與 target 比較。
#ifndef AMNT_HOST_TEST
typedef unsigned long long u64;
typedef unsigned int u32;
#define AMNT_INLINE __device__ __forceinline__
#define AMNT_KERNEL extern "C" __global__
#endif

#define ROTL64(x, y) (((x) << (y)) | ((x) >> (64 - (y))))

__constant__ u64 AMNT_RC[24] = {
  0x0000000000000001ULL, 0x0000000000008082ULL, 0x800000000000808aULL, 0x8000000080008000ULL,
  0x000000000000808bULL, 0x0000000080000001ULL, 0x8000000080008081ULL, 0x8000000000008009ULL,
  0x000000000000008aULL, 0x0000000000000088ULL, 0x0000000080008009ULL, 0x000000008000000aULL,
  0x000000008000808bULL, 0x800000000000008bULL, 0x8000000000008089ULL, 0x8000000000008003ULL,
  0x8000000000008002ULL, 0x8000000000000080ULL, 0x000000000000800aULL, 0x800000008000000aULL,
  0x8000000080008081ULL, 0x8000000000008080ULL, 0x0000000080000001ULL, 0x8000000080008008ULL};

AMNT_INLINE u64 amnt_bswap64(u64 x) {
  x = ((x & 0x00000000FFFFFFFFULL) << 32) | ((x & 0xFFFFFFFF00000000ULL) >> 32);
  x = ((x & 0x0000FFFF0000FFFFULL) << 16) | ((x & 0xFFFF0000FFFF0000ULL) >> 16);
  x = ((x & 0x00FF00FF00FF00FFULL) << 8)  | ((x & 0xFF00FF00FF00FF00ULL) >> 8);
  return x;
}

AMNT_INLINE void amnt_keccakf(u64 st[25]) {
  const int rotc[24] = {1, 3, 6, 10, 15, 21, 28, 36, 45, 55, 2, 14, 27, 41, 56, 8, 25, 43, 62, 18, 39, 61, 20, 44};
  const int piln[24] = {10, 7, 11, 17, 18, 3, 5, 16, 8, 21, 24, 4, 15, 23, 19, 13, 12, 2, 20, 14, 22, 9, 6, 1};
  u64 bc[5], t;
#pragma unroll
  for (int r = 0; r < 24; r++) {
#pragma unroll
    for (int i = 0; i < 5; i++) bc[i] = st[i] ^ st[i + 5] ^ st[i + 10] ^ st[i + 15] ^ st[i + 20];
#pragma unroll
    for (int i = 0; i < 5; i++) {
      t = bc[(i + 4) % 5] ^ ROTL64(bc[(i + 1) % 5], 1);
#pragma unroll
      for (int j = 0; j < 25; j += 5) st[j + i] ^= t;
    }
    t = st[1];
#pragma unroll
    for (int i = 0; i < 24; i++) {
      int j = piln[i];
      bc[0] = st[j];
      st[j] = ROTL64(t, rotc[i]);
      t = bc[0];
    }
#pragma unroll
    for (int j = 0; j < 25; j += 5) {
#pragma unroll
      for (int i = 0; i < 5; i++) bc[i] = st[j + i];
#pragma unroll
      for (int i = 0; i < 5; i++) st[j + i] ^= (~bc[(i + 1) % 5]) & bc[(i + 2) % 5];
    }
    st[0] ^= AMNT_RC[r];
  }
}

// tgt[0..3]：target 的 4 個 64-bit 字（大端序，tgt[0] 為最高位）。
// 找到 digest <= target 時，把計數器 ctr 寫入 out。nonce = (prefix << 64) | ctr。
AMNT_KERNEL void amnt_search(const u64* __restrict__ pre, const u64* __restrict__ tgt,
                                       u64 start, u32 iters, u64* out, u32* out_count, u32 max_out) {
  u64 tid = (u64)blockIdx.x * blockDim.x + threadIdx.x;
  u64 stride = (u64)gridDim.x * blockDim.x;
  u64 base[25];
#pragma unroll
  for (int i = 0; i < 25; i++) base[i] = pre[i];
  u64 t0 = tgt[0], t1 = tgt[1], t2 = tgt[2], t3 = tgt[3];
  for (u32 it = 0; it < iters; it++) {
    u64 ctr = start + (u64)it * stride + tid;
    u64 st[25];
#pragma unroll
    for (int i = 0; i < 25; i++) st[i] = base[i];
    st[6] ^= amnt_bswap64(ctr);
    amnt_keccakf(st);
    u64 d = amnt_bswap64(st[0]);
    if (d > t0) continue;
    if (d == t0) {
      d = amnt_bswap64(st[1]);
      if (d > t1) continue;
      if (d == t1) {
        d = amnt_bswap64(st[2]);
        if (d > t2) continue;
        if (d == t2 && amnt_bswap64(st[3]) > t3) continue;
      }
    }
    u32 k = atomicAdd(out_count, 1u);
    if (k < max_out) out[k] = ctr;
  }
}

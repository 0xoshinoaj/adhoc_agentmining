// 在沒有 GPU 的機器上，用 gcc 把 keccak_kernel.cu 當 C 編譯，序列模擬整個 grid，驗證 kernel 邏輯。
#include <stdio.h>
#include <stdlib.h>
typedef unsigned long long u64;
typedef unsigned int u32;
#define AMNT_HOST_TEST 1
#define AMNT_INLINE static inline
#define __constant__ static const
#define __global__
#define AMNT_KERNEL
#define __restrict__
#define __device__
struct dim { u32 x; };
static struct dim blockIdx, threadIdx, blockDim, gridDim;
static u32 atomicAdd(u32* p, u32 v) { u32 o = *p; *p += v; return o; }
#include "../keccak_kernel.cu"
// stdin：25 個 pre lane、4 個 target 字、start、iters、grid、block（皆十進位）
int main(void) {
  u64 pre[25], tgt[4], start; u32 iters, g, b;
  for (int i = 0; i < 25; i++) if (scanf("%llu", &pre[i]) != 1) return 1;
  for (int i = 0; i < 4; i++) if (scanf("%llu", &tgt[i]) != 1) return 1;
  if (scanf("%llu %u %u %u", &start, &iters, &g, &b) != 4) return 1;
  u64 out[4096]; u32 cnt = 0;
  gridDim.x = g; blockDim.x = b;
  for (u32 bi = 0; bi < g; bi++) for (u32 ti = 0; ti < b; ti++) {
    blockIdx.x = bi; threadIdx.x = ti;
    amnt_search(pre, tgt, start, iters, out, &cnt, 4096);
  }
  for (u32 i = 0; i < cnt && i < 4096; i++) printf("%llu\n", out[i]);
  return 0;
}

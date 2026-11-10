"""單元測試（CPU）：python tests/test_all.py   需要網路的項目會讀公開 BSC RPC（唯讀）。"""
import os, random, secrets, subprocess, sys, tempfile, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import amnt_core as C

ROOT = os.path.join(os.path.dirname(__file__), "..")
ok = 0


def check(cond, name):
    global ok
    if not cond:
        raise SystemExit(f"FAIL: {name}")
    ok += 1
    print("PASS", name)


def sponge(msg):  # 用純 Python keccakf 實作完整 keccak256，對照 pycryptodome
    rate = 136
    m = bytearray(msg) + b"\x01"
    m += bytes((-len(m)) % rate)
    m[-1] |= 0x80
    st = [0] * 25
    for o in range(0, len(m), rate):
        for i in range(17):
            st[i] ^= int.from_bytes(m[o + 8 * i:o + 8 * i + 8], "little")
        C.keccakf(st)
    return b"".join(x.to_bytes(8, "little") for x in st[:4])


# 1) keccak-f 正確性
for n in [0, 1, 135, 136, 137, 192, 300]:
    d = secrets.token_bytes(n)
    check(sponge(d) == C.keccak256(d), f"keccakf 純 Python == pycryptodome（{n} bytes）")
check(C.keccak256(b"").hex() == "c5d2460186f7233c927e7db2dcc703c0e500b653ca82273b7bfad8045d85a470", "keccak256('') 已知值")

# 2) 中間狀態（GPU 演算法）== 完整 abi.encode 雜湊
seed = secrets.token_bytes(32)
aid = random.getrandbits(256)
pre = C.build_prestate(seed, C.DEFAULT_WALLET, aid, 1, 3, prefix := random.getrandbits(192))
for _ in range(5):
    ctr = random.getrandbits(64)
    check(C.digest_from_prestate(pre, ctr) == C.proof_digest(seed, C.DEFAULT_WALLET, aid, 1, 3, (prefix << 64) | ctr),
          "prestate + lane6 計數器 == keccak256(abi.encode(...))")

# 3) 與合約 proofDigest() 比對（eth_call，唯讀）
ch = C.Chain()
st = ch.state()
real_aid = ch.agent_id(C.DEFAULT_WALLET)
for _ in range(3):
    nonce = random.getrandbits(256)
    check(ch.chain_digest(st["day"], C.DEFAULT_WALLET, real_aid, 1, nonce) ==
          C.proof_digest(st["seed"], C.DEFAULT_WALLET, real_aid, 1, st["day"], nonce),
          f"本地 digest == 合約 proofDigest()（day {st['day']}，真實種子）")

# 4) calldata：用 eth_call 模擬 mineBatch，確認合約能解碼（預期因未選方案/無效解答 revert，而非解碼失敗）
data = C.calldata_mine_batch(1, [random.getrandbits(256) for _ in range(3)])
try:
    ch._rpc("eth_call", [{"from": C.DEFAULT_WALLET, "to": C.AMNT, "data": data}, "latest"])
    check(False, "mineBatch 應 revert")
except RuntimeError as e:
    msg = str(e)
    errs = {C.selector(x).hex(): x for x in ["WrongLane()", "InvalidProof()", "NoSession()", "SessionNotPinned()", "GpuDailyCapExceeded()"]}
    hit = [v for k, v in errs.items() if k in msg]
    print("   mineBatch 模擬 revert：", hit or msg[:200])
    check(bool(hit), "mineBatch calldata 可被合約解碼（回傳合約自訂錯誤）")
check(C.calldata_select_plan(1).startswith("0x" + C.selector("selectPlan(uint8)").hex()), "selectPlan calldata")

# 5) CUDA kernel 原始碼用 gcc 編成 CPU 版本，低難度下與 Python 暴力搜尋結果一致
exe = os.path.join(tempfile.gettempdir(), "amnt_harness")
subprocess.check_call(["gcc", "-O2", "-o", exe, os.path.join(ROOT, "tests", "kernel_cpu_harness.c")])
bits = 10
target = (1 << (256 - bits)) - 1
start, iters, g, b = random.getrandbits(40), 32, 8, 64
inp = " ".join(map(str, pre + C.target_words(target) + [start, iters, g, b]))
got = sorted(int(x) for x in subprocess.run([exe], input=inp, capture_output=True, text=True, check=True).stdout.split())
n = iters * g * b
expect = sorted(c for c in range(start, start + n) if C.proof_digest(seed, C.DEFAULT_WALLET, aid, 1, 3, (prefix << 64) | c) <= target)
print(f"   kernel(CPU 模擬) 找到 {len(got)} 個，Python 找到 {len(expect)} 個（共試 {n} 個，{bits} bit）")
check(got == expect and len(got) > 0, "keccak_kernel.cu 邏輯 == Python 參考實作")
# 邊界：target 的低位字比較（target 高 64 bit 與 digest 相同時）
c0 = expect[0]
d0 = C.proof_digest(seed, C.DEFAULT_WALLET, aid, 1, 3, (prefix << 64) | c0)
for tgt, should in [(d0, True), (d0 - 1, False)]:
    inp = " ".join(map(str, pre + C.target_words(tgt) + [c0, 1, 1, 1]))
    r = subprocess.run([exe], input=inp, capture_output=True, text=True, check=True).stdout.split()
    check((str(c0) in r) == should, f"kernel 邊界比較 digest {'==' if should else '>'} target")

# 6) 端到端：CPU 模式 offline 低難度挖礦，輸出檔格式
with tempfile.TemporaryDirectory() as td:
    subprocess.check_call([sys.executable, os.path.join(ROOT, "amnt_miner.py"), "mine", "--cpu", "--offline", "--test-bits", "12",
                           "--max-solutions", "12", "--out-dir", td, "--flush-secs", "0"], stdout=subprocess.DEVNULL)
    f = [x for x in os.listdir(td) if x.startswith("solutions_")][0]
    d = json.load(open(os.path.join(td, f), encoding="utf-8"))
    sd = bytes.fromhex(d["seed"][2:])
    allgood = all(C.proof_digest(sd, d["wallet"], int(d["agentId"]), 1, d["day"], int(nn)) == int(h, 16) <= (1 << 244)
                  for bb in d["batches"] for nn, h in zip(bb["nonces"], bb["hashes"]))
    sizes = [len(bb["nonces"]) for bb in d["batches"]]
    print("   batch 大小：", sizes)
    check(sizes[0] == 10 and max(sizes) <= 10 and sum(sizes) >= 12 and allgood, "solutions 檔：每批 ≤10、hash 皆可重算且 ≤ target")
    check(d["batches"][0]["calldata"] == C.calldata_mine_batch(1, [int(x) for x in d["batches"][0]["nonces"]]), "batch calldata 一致")

print(f"\n全部通過：{ok} 項")

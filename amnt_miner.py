#!/usr/bin/env python3
"""AMNT Plan A 挖礦解答計算器（只計算，不送交易）。

指令：
  python amnt_miner.py status            讀鏈上狀態（唯讀）
  python amnt_miner.py mine [--cpu]      搜尋解答，寫入 solutions_YYYYMMDD.json
  python amnt_miner.py check FILE        依「當下」鏈上狀態重新驗證，輸出可送出的 ready_*.json

安全：沒有私鑰、助記詞、簽名或送交易功能；只用公開 BSC RPC 做 eth_call。
"""
import argparse
import datetime as dt
import json
import os
import secrets
import sys
import time

import amnt_core as C

HERE = os.path.dirname(os.path.abspath(__file__))


def log(msg):
    print(f"[{dt.datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def fmt_ts(ts):
    return dt.datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")


# ---------------- 搜尋後端 ----------------
class CpuSearcher:
    """CPU 後備模式（慢，只適合測試或低難度）：直接對完整 192 bytes 做 keccak256。"""
    name = "CPU"

    def __init__(self, ctx):
        self.set_job(ctx)

    def set_job(self, ctx):
        self.fixed = C.proof_preimage(ctx["seed"], ctx["wallet"], ctx["agentId"], ctx["lane"], ctx["day"], 0)[:160]
        self.prefix = ctx["prefix"]
        self.target = ctx["target"]

    def set_target(self, t):
        self.target = t

    def search(self, start, count):
        found = []
        hi = self.prefix << 64
        for ctr in range(start, start + count):
            d = int.from_bytes(C.keccak256(self.fixed + C.word(hi | ctr)), "big")
            if d <= self.target:
                found.append(ctr)
        return found, count


class GpuSearcher:
    """CUDA（CuPy RawKernel）。kernel 原始碼在 keccak_kernel.cu。"""
    name = "GPU"

    def __init__(self, ctx, threads=256, blocks=None, target_secs=0.3):
        import cupy as cp
        self.cp = cp
        src = open(os.path.join(HERE, "keccak_kernel.cu"), encoding="utf-8").read()
        self.kernel = cp.RawKernel(src, "amnt_search")
        dev = cp.cuda.Device()
        sms = dev.attributes["MultiProcessorCount"]
        self.threads = threads
        self.dev_blocks = sms * 32
        self.blocks = blocks or self.dev_blocks
        self.iters = 16
        self.target_secs = target_secs
        self.max_out = 1024
        self.out = cp.zeros(self.max_out, dtype=cp.uint64)
        self.cnt = cp.zeros(1, dtype=cp.uint32)
        props = cp.cuda.runtime.getDeviceProperties(dev.id)
        log(f"GPU：{props['name'].decode(errors='ignore')}，SM={sms}，grid={self.blocks}x{self.threads}")
        self.set_job(ctx)

    def set_job(self, ctx):
        cp = self.cp
        import numpy as np
        pre = C.build_prestate(ctx["seed"], ctx["wallet"], ctx["agentId"], ctx["lane"], ctx["day"], ctx["prefix"])
        self.pre = cp.asarray(np.array(pre, dtype=np.uint64))
        self.set_target(ctx["target"])

    def set_target(self, t):
        import numpy as np
        self.tgt = self.cp.asarray(np.array(C.target_words(t), dtype=np.uint64))

    def batch_size(self):
        return self.threads * self.blocks * self.iters

    def search(self, start, count_hint=None):
        import numpy as np
        cp = self.cp
        self.cnt.fill(0)
        n = self.batch_size()
        t0 = time.time()
        self.kernel((self.blocks,), (self.threads,),
                    (self.pre, self.tgt, np.uint64(start), np.uint32(self.iters), self.out, self.cnt, np.uint32(self.max_out)))
        cp.cuda.Stream.null.synchronize()
        el = time.time() - t0
        k = int(self.cnt.get()[0])
        found = [int(x) for x in self.out[:min(k, self.max_out)].get()]
        # 自動調整每次啟動的工作量，讓單次約 target_secs 秒
        if el > 0:
            self.iters = max(1, min(4096, int(self.iters * self.target_secs / el)))
        return found, n


# ---------------- 輸出檔 ----------------
class SolutionFile:
    def __init__(self, out_dir, ctx, session_start_ts):
        date = dt.datetime.fromtimestamp(session_start_ts).strftime("%Y%m%d")
        self.path = os.path.join(out_dir, f"solutions_{date}.json")
        if os.path.exists(self.path):
            self.data = json.load(open(self.path, encoding="utf-8"))
            if self.data.get("day") != ctx["day"] or self.data.get("agentId") != str(ctx["agentId"]):
                self.path = self.path.replace(".json", f"_day{ctx['day']}_{int(time.time())}.json")
                self.data = None
        else:
            self.data = None
        if self.data is None:
            self.data = {
                "說明": "每個 batch 可直接組成 mineBatch(lane, nonces[])；送出前請先跑 check 或 make_baw_cmds.py 依當下 target 重新過濾。",
                "contract": C.AMNT, "chainId": C.CHAIN_ID, "function": "mineBatch(uint8,uint256[])", "value": "0",
                "wallet": ctx["wallet"], "agentId": str(ctx["agentId"]), "lane": ctx["lane"], "day": ctx["day"],
                "seed": "0x" + ctx["seed"].hex(), "batches": [],
            }

    def add_batch(self, sols, target):
        b = {
            "batch": len(self.data["batches"]) + 1,
            "createdAt": fmt_ts(time.time()),
            "targetAtFind": hex(target),
            "nonces": [str(s["nonce"]) for s in sols],
            "hashes": ["0x%064x" % s["digest"] for s in sols],
            "allDaySafe": [s["allDaySafe"] for s in sols],
            "to": C.AMNT, "value": "0",
            "calldata": C.calldata_mine_batch(self.data["lane"], [s["nonce"] for s in sols]),
        }
        self.data["batches"].append(b)
        tmp = self.path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self.data, f, ensure_ascii=False, indent=2)
        os.replace(tmp, self.path)
        log(f"已寫入 batch #{b['batch']}（{len(sols)} 個解答）→ {self.path}")


# ---------------- 子指令 ----------------
def resolve_ids(chain, args):
    wallet = args.wallet
    agent_id = int(args.agent_id) if args.agent_id else chain.agent_id(wallet)
    if agent_id == 0:
        sys.exit("這個錢包沒有 Agent ID，無法挖礦。")
    return wallet, agent_id


def cmd_status(args):
    ch = C.Chain()
    wallet, aid = resolve_ids(ch, args)
    s = ch.state(aid)
    lane_txt = {0: "尚未選擇（需在時段內送 selectPlan(1)，value 0）", 1: "Plan A", 2: "Plan B（不是 Plan A！）"}.get(s["lane"], s["lane"])
    print(f"錢包 {wallet}\nAgent ID {aid}\n目前方案 {lane_txt}")
    print(f"第 {s['day']} 天，時段開放：{s['inSession']}，本時段結束 {fmt_ts(s['sessionEnd'])}")
    print(f"種子 {'0x' + s['seed'].hex() if any(s['seed']) else '尚未固定 → 需先 pinSession()'}")
    print(f"gpuTarget {hex(s['gpuTarget'])}（約 {C.bits_of(s['gpuTarget']):.3f} bit；範圍 {C.bits_of(s['gpuTargetMax']):.0f}–{C.bits_of(s['gpuTargetMin']):.0f} bit）")
    print(f"全網今日 Plan A {s['gpuProofsToday']}/{s['gpuDailyTarget']}，距下次調難度還差 {s['retargetInterval'] - s['gpuSinceRetarget']} 個")
    print(f"每個解答 {s['gpuReward'] / 1e8:,.0f} AMNT")


def cmd_mine(args):
    offline = args.offline
    ch = None if offline else C.Chain()
    if offline:
        wallet, aid = args.wallet, int(args.agent_id or 1)
        s = {"day": 0, "seed": secrets.token_bytes(32), "inSession": True, "gpuTarget": 0, "gpuTargetMin": 0,
             "gpuProofsToday": 0, "gpuDailyTarget": 1 << 60, "sessionEnd": time.time() + 10 ** 6,
             "anchor": int(time.time()), "lane": 1}
    else:
        wallet, aid = resolve_ids(ch, args)
        while True:
            s = ch.state(aid)
            if not s["inSession"]:
                nxt = s["anchor"] + (s["day"] + (1 if s["now"] >= s["sessionEnd"] else 0)) * s["dayLength"]
                msg = f"目前不在挖礦時段，下次開放約 {fmt_ts(nxt)}（本機時間）。"
                if not args.wait:
                    sys.exit(msg + " 加上 --wait 可自動等待。")
                log(msg + " 等待中…")
                time.sleep(min(60, max(5, nxt - s["now"])))
                continue
            if not any(s["seed"]):
                log("今日種子尚未固定：需先 pinSession()（請 agent 依 README 送出，value 0）。程式不會自己送，等待中…")
                time.sleep(args.poll)
                continue
            if s["lane"] == 2 and not args.force:
                sys.exit("這個 Agent ID 目前是 Plan B，不能挖 Plan A。換方案要 0.01 BNB，請先跟使用者確認。")
            if s["lane"] == 0:
                log("警告：尚未選方案。解答要能送出，必須先在時段內送 selectPlan(1)（value 0、只限第一次）。")
            if s["gpuProofsToday"] >= s["gpuDailyTarget"]:
                sys.exit("全網今日 Plan A 上限已滿，明天再來。")
            break

    def target_now(st):
        if args.test_bits:
            return (1 << int(256 - args.test_bits)) - 1
        return st["gpuTarget"]

    ctx = {"seed": s["seed"], "wallet": wallet, "agentId": aid, "lane": C.LANE_A, "day": s["day"],
           "prefix": secrets.randbits(192), "target": target_now(s)}
    searcher = CpuSearcher(ctx) if args.cpu else GpuSearcher(ctx)
    out = SolutionFile(args.out_dir, ctx, s["anchor"] + s["day"] * s.get("dayLength", 86400))
    log(f"開始搜尋（{searcher.name}），day={ctx['day']}，Agent ID={aid}，target≈{C.bits_of(ctx['target']):.3f} bit")

    pending, ctr, total = [], 0, 0
    t_start = last_poll = last_print = time.time()
    rate_t, rate_n = time.time(), 0
    found_total = 0
    min_target = s["gpuTargetMin"]

    def flush(reason):
        nonlocal pending
        while pending:
            chunk, pending = pending[:args.batch_size], pending[args.batch_size:]
            out.add_batch(chunk, ctx["target"])
        if reason:
            log(reason)

    while True:
        found, n = searcher.search(ctr, args.cpu_chunk) if args.cpu else searcher.search(ctr)
        ctr += n
        total += n
        rate_n += n
        for c in found:
            nonce = (ctx["prefix"] << 64) | c
            d = C.proof_digest(ctx["seed"], wallet, aid, ctx["lane"], ctx["day"], nonce)  # CPU 重算驗證
            if d > ctx["target"]:
                log(f"警告：GPU 回報的 nonce CPU 驗證不通過，已丟棄（ctr={c}）")
                continue
            pending.append({"nonce": nonce, "digest": d, "allDaySafe": bool(min_target and d <= min_target), "t": time.time()})
            found_total += 1
            log(f"找到解答 #{found_total}：hash=0x{d:064x}")
        if len(pending) >= args.batch_size:
            flush(None)
        elif pending and args.flush_secs and time.time() - pending[0]["t"] >= args.flush_secs:
            flush(f"解答已等 {args.flush_secs} 秒，先輸出不足 10 個的 batch。")

        now = time.time()
        if now - last_print >= 5:
            rate = rate_n / max(1e-9, now - rate_t)
            exp = 2 ** C.bits_of(ctx["target"]) / rate if rate else float("inf")
            log(f"速度 {rate / 1e6:,.1f} MH/s，目前難度 {C.bits_of(ctx['target']):.3f} bit，平均每解約 {exp:,.0f} 秒，已找到 {found_total}")
            last_print, rate_t, rate_n = now, now, 0
        if args.max_seconds and now - t_start >= args.max_seconds:
            flush("到達 --max-seconds，結束。")
            return
        if args.max_solutions and found_total >= args.max_solutions:
            flush("到達 --max-solutions，結束。")
            return
        if not offline and now - last_poll >= args.poll:
            last_poll = now
            try:
                st = ch.state(aid)
            except Exception as e:
                log(f"讀鏈失敗（稍後重試）：{e}")
                continue
            if st["day"] != ctx["day"] or not st["inSession"] or st["seed"] != ctx["seed"]:
                flush("時段結束或日期／種子改變，停止。舊解答到下一個時段都會失效。")
                return
            if st["gpuProofsToday"] >= st["gpuDailyTarget"]:
                flush("全網今日 Plan A 上限已滿，停止。")
                return
            nt = target_now(st)
            if nt != ctx["target"]:
                log(f"難度調整：{C.bits_of(ctx['target']):.3f} → {C.bits_of(nt):.3f} bit（全網今日 {st['gpuProofsToday']}）")
                before = len(pending)
                pending = [p for p in pending if p["digest"] <= nt]
                if len(pending) != before:
                    log(f"丟棄 {before - len(pending)} 個高於新 target 的待輸出解答")
                ctx["target"] = nt
                searcher.set_target(nt)


def cmd_check(args):
    ch = C.Chain()
    data = json.load(open(args.file, encoding="utf-8"))
    aid, wallet, lane, day = int(data["agentId"]), data["wallet"], int(data["lane"]), int(data["day"])
    seed = bytes.fromhex(data["seed"][2:])
    s = ch.state(aid)
    problems = []
    if s["day"] != day or s["seed"] != seed:
        problems.append("日期或種子已改變：這個檔案的解答全部失效。")
    if not s["inSession"]:
        problems.append("目前不在挖礦時段：mineBatch 會 revert。")
    if s["lane"] != lane:
        problems.append(f"這個 Agent ID 目前方案是 {s['lane']}，與解答的 lane {lane} 不符（0 = 尚未 selectPlan）。")
    if s["gpuProofsToday"] >= s["gpuDailyTarget"]:
        problems.append("全網今日 Plan A 上限已滿。")
    good, bad = [], 0
    for b in data["batches"]:
        for nonce_s, h in zip(b["nonces"], b["hashes"]):
            nonce, d = int(nonce_s), int(h, 16)
            local = C.proof_digest(seed, wallet, aid, lane, day, nonce)
            if local != d:
                bad += 1
                continue
            if s["day"] == day and s["seed"] == seed:
                if args.verify_onchain and ch.chain_digest(day, wallet, aid, lane, nonce) != d:
                    bad += 1
                    continue
                if d > s["gpuTarget"] or ch.spent(d):
                    bad += 1
                    continue
            good.append((nonce, d))
    good.sort(key=lambda x: x[1])  # hash 越小越安全，排前面
    batches = []
    for i in range(0, len(good), 10):
        chunk = good[i:i + 10]
        batches.append({"nonces": [str(n) for n, _ in chunk], "hashes": ["0x%064x" % d for _, d in chunk],
                        "to": C.AMNT, "value": "0", "calldata": C.calldata_mine_batch(lane, [n for n, _ in chunk])})
    res = {"checkedAt": fmt_ts(time.time()), "currentTarget": hex(s["gpuTarget"]), "contract": C.AMNT, "chainId": C.CHAIN_ID,
           "wallet": wallet, "lane": lane, "day": day, "problems": problems, "valid": len(good), "dropped": bad,
           "batches": batches if not problems else []}
    path = os.path.join(os.path.dirname(os.path.abspath(args.file)), f"ready_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    json.dump(res, open(path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"有效 {len(good)}、失效 {bad}；{'問題：' + '；'.join(problems) if problems else '可以送出'}\n→ {path}")
    return res, path


def cmd_selftest(args):
    """GPU 自我測試（不需網路）：低難度下，GPU kernel 結果必須和 CPU 逐一重算完全一致。"""
    ctx = {"seed": secrets.token_bytes(32), "wallet": args.wallet, "agentId": secrets.randbits(256), "lane": C.LANE_A,
           "day": 0, "prefix": secrets.randbits(192), "target": (1 << (256 - 12)) - 1}
    g = GpuSearcher(ctx, threads=256, blocks=64)
    g.iters = 4
    start = secrets.randbits(40)
    found, n = g.search(start)
    cpu = CpuSearcher(ctx)
    expect, _ = cpu.search(start, n)
    print(f"GPU 找到 {len(found)} 個，CPU 找到 {len(expect)} 個（共 {n} 個 nonce，12 bit）")
    if sorted(found) != sorted(expect) or not expect:
        sys.exit("自我測試失敗：GPU 與 CPU 結果不一致，請勿使用 GPU 模式，回報給我們。")
    g.iters = 64
    g.blocks = g.dev_blocks
    t0 = time.time(); tot = 0
    while time.time() - t0 < 5:
        _, k = g.search(secrets.randbits(40)); tot += k
    rate = tot / (time.time() - t0)
    print(f"自我測試通過。粗估速度 {rate / 1e6:,.0f} MH/s；Plan A 35 bit 平均約 {2 ** 35 / rate:,.0f} 秒、38 bit 約 {2 ** 38 / rate:,.0f} 秒一個解答。")


def main():
    p = argparse.ArgumentParser(description="AMNT Plan A 解答計算器（只計算，不送交易）")
    p.add_argument("--wallet", default=C.DEFAULT_WALLET)
    p.add_argument("--agent-id", help="不填則從鏈上查")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("status")
    m = sub.add_parser("mine")
    m.add_argument("--cpu", action="store_true", help="CPU 後備模式（很慢，測試用）")
    m.add_argument("--cpu-chunk", type=int, default=20000)
    m.add_argument("--out-dir", default=".")
    m.add_argument("--batch-size", type=int, default=10)
    m.add_argument("--flush-secs", type=int, default=120, help="解答等超過幾秒就先輸出不足 10 個的 batch（0 = 只滿 10 才輸出）")
    m.add_argument("--poll", type=int, default=6, help="每幾秒讀一次鏈上狀態")
    m.add_argument("--wait", action="store_true", help="不在時段時等待開放")
    m.add_argument("--force", action="store_true")
    m.add_argument("--test-bits", type=int, help="測試用：改用指定難度（找到的解答送上鏈會失敗）")
    m.add_argument("--offline", action="store_true", help="測試用：不讀鏈、隨機種子")
    m.add_argument("--max-seconds", type=int, default=0)
    m.add_argument("--max-solutions", type=int, default=0)
    sub.add_parser("selftest", help="GPU 與 CPU 結果比對（第一次使用必跑）")
    c = sub.add_parser("check")
    c.add_argument("file")
    c.add_argument("--verify-onchain", action="store_true", help="每個 nonce 再用合約 proofDigest() 驗一次")
    args = p.parse_args()
    {"status": cmd_status, "mine": cmd_mine, "check": cmd_check, "selftest": cmd_selftest}[args.cmd](args)


if __name__ == "__main__":
    main()

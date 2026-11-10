#!/usr/bin/env python3
"""只「產生」baw contract-call 指令字串給 agent 執行，本腳本不會送出任何交易、也不呼叫 baw。

用法：
  python make_baw_cmds.py solutions_20261111.json   先依當下鏈上狀態重新過濾（同 amnt_miner.py check），再印出指令
  python make_baw_cmds.py --setup                    只印出需要的 selectPlan(1) / pinSession() 指令（依鏈上狀態判斷）
"""
import argparse
import types

import amnt_core as C
import amnt_miner as M


def preview_cmd(wallet, data):
    return (f"baw contract-call preview --binanceChainId {C.CHAIN_ID} --from {wallet} "
            f"--to {C.AMNT} --value 0 --inputData {data} --json")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("file", nargs="?")
    p.add_argument("--wallet", default=C.DEFAULT_WALLET)
    p.add_argument("--setup", action="store_true")
    a = p.parse_args()
    ch = C.Chain()
    aid = ch.agent_id(a.wallet)
    s = ch.state(aid)
    print("# 規則：只送下列指令；對象只能是 AMNT 合約；value 一律 0；preview 失敗（revert）就停，不要重試、不要改參數。")
    print("# 每個 preview 成功後，才用回傳的 requestId 執行：baw contract-call execute --requestId <requestId> --json\n")
    if s["lane"] == 0:
        if s["inSession"]:
            print("# 1) 第一次選方案：Plan A（lane 1），免費，只能在時段內，一生只送一次")
            print(preview_cmd(a.wallet, C.calldata_select_plan(C.LANE_A)) + "\n")
        else:
            print("# 尚未選方案，但目前不在時段：等時段開放再送 selectPlan(1)。時段外送 selectPlan 會 revert（FirstPickOutsideWindow）；也絕對不要帶 BNB（帶 0.01 BNB 是換方案）。\n")
    elif s["lane"] != C.LANE_A:
        print(f"# 停止：這個 Agent ID 目前是 lane {s['lane']}（不是 Plan A）。換方案要 0.01 BNB，須使用者另外同意，本腳本不產生。")
        return
    if s["inSession"] and not any(s["seed"]):
        print("# 2) 今日種子尚未固定：送 pinSession()（任何人都可送，value 0）")
        print(preview_cmd(a.wallet, C.calldata_pin_session()) + "\n")
    if a.setup or not a.file:
        return
    res, path = M.cmd_check(types.SimpleNamespace(file=a.file, verify_onchain=True))
    if res["problems"]:
        print("# 不產生 mineBatch 指令：" + "；".join(res["problems"]))
        return
    for i, b in enumerate(res["batches"], 1):
        print(f"# mineBatch 第 {i} 筆：{len(b['nonces'])} 個解答（約 {len(b['nonces']) * 1500:,} AMNT）")
        print(preview_cmd(a.wallet, b["calldata"]) + "\n")
    if not res["batches"]:
        print("# 沒有仍然有效的解答。")


if __name__ == "__main__":
    main()

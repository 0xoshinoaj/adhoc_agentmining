# AMNT Plan A 解答計算器（Windows + NVIDIA GPU）

這個程式只負責**算出** Agent Mining（AMNT）Plan A 的挖礦解答，寫成 JSON 檔。送交易這一步，由這台電腦上的 agent 用 BAW 錢包（`baw contract-call`）執行。

- 合約：`0x2d07B9CcDF5FCdbE26167358B94eeb4c9E995e4E`（BSC，chainId 56，Sourcify exact match）
- Plan A = lane `1`，每個解答 1,500 AMNT，每筆交易最多 10 個解答
- 挖礦時段：每天 21:00–23:00（台北時間），共 45 天，第 0 天是 2026-11-10

## 安全說明

- 程式裡**沒有**私鑰、助記詞、簽名或送交易的功能。BAW 是 MPC 錢包，本來就沒有私鑰。
- 只對公開 BSC RPC 做唯讀查詢（`eth_call`、`eth_getBlockByNumber`），不需要 API key。
- `make_baw_cmds.py` 只會**印出**指令字串，不會呼叫 baw，也不會送出任何東西。
- 不會安裝或執行任何網站提供的腳本，不會連第三方 MCP。

## 檔案

| 檔案 | 用途 |
|---|---|
| `amnt_miner.py` | 主程式：`status`／`selftest`／`mine`／`check` |
| `amnt_core.py` | keccak、abi.encode、唯讀 RPC、calldata 產生 |
| `keccak_kernel.cu` | CUDA keccak-f[1600] 搜尋 kernel（CuPy 在執行時編譯） |
| `make_baw_cmds.py` | 依鏈上狀態產生 `baw contract-call preview` 指令字串（不送出） |
| `run_windows.bat` | 一鍵：建環境 → 自我測試 → 顯示狀態 → 開始挖（時段外自動等待） |
| `tests/` | CPU 單元測試（含把 CUDA kernel 用 gcc 編成 CPU 版比對） |

## Windows 安裝

1. 更新 NVIDIA 顯示卡驅動到支援 CUDA 12 的版本（Game Ready 或 Studio 驅動 ≥ 528 即可）。在命令提示字元輸入 `nvidia-smi`，右上角 CUDA Version 顯示 12.x 以上就可以。
2. 安裝 Python 3.11–3.13（64 位元）：https://www.python.org/downloads/windows/ 安裝時勾選「Add python.exe to PATH」。
3. 在程式資料夾開啟命令提示字元：
   ```bat
   py -3 -m venv .venv
   .venv\Scripts\python -m pip install --upgrade pip
   .venv\Scripts\pip install -r requirements.txt
   ```
   `cupy-cuda12x[ctk]` 會一併安裝需要的 CUDA 元件，**不必**另外安裝 CUDA Toolkit。
   如果之後出現 `vector_types.h` 找不到的錯誤，再執行：`.venv\Scripts\pip install "nvidia-cuda-runtime-cu12==12.*"`。
4. 也可以直接雙擊 `run_windows.bat`，它會自動做第 3 步並開始挖。

## 執行

```bat
.venv\Scripts\python amnt_miner.py selftest     REM 第一次必跑：GPU 與 CPU 結果比對，並測速度
.venv\Scripts\python amnt_miner.py status       REM 鏈上狀態：Agent ID、方案、種子、難度、全網用量
.venv\Scripts\python amnt_miner.py mine --wait  REM 開始搜尋；時段外會等到 21:00
```

`mine` 常用參數：

- `--flush-secs 120`（預設）：解答等超過 120 秒，就先輸出不足 10 個的 batch。解答放越久越可能失效，詳見下方「解答時效」。設成 `0` 就只在湊滿 10 個時輸出。
- `--poll 6`：每 6 秒讀一次鏈上狀態。難度改了就換新 target；日期、種子改變或時段結束就停止。
- `--cpu`：CPU 後備模式，非常慢，只用來測試。
- `--wallet`、`--agent-id`：預設錢包是 `0xaa4d88F35Efd271B65Deee9B8EAeA4D66C897597`，Agent ID 會從鏈上自動查。

畫面每 5 秒會顯示速度（MH/s）、目前難度（bit）和平均每解要幾秒。

## 輸出檔

`solutions_YYYYMMDD.json`，日期是當天時段的本機日期：

```json
{
  "contract": "0x2d07…4E", "chainId": 56, "function": "mineBatch(uint8,uint256[])", "value": "0",
  "wallet": "0xaa4d…7597", "agentId": "5362…", "lane": 1, "day": 0, "seed": "0x…",
  "batches": [
    { "batch": 1, "nonces": ["…10 個十進位字串…"], "hashes": ["0x…"], "allDaySafe": [true, false, …],
      "targetAtFind": "0x…", "to": "0x2d07…4E", "value": "0", "calldata": "0x738c12e5…" }
  ]
}
```

- `calldata` 已經是完整的 `mineBatch(1, nonces[])` 編碼，可以直接用在 `baw contract-call preview --inputData`。
- `allDaySafe = true` 表示這個 hash 也低於當天最難的 target（38 bit），整個時段內不管難度怎麼調都不會失效。

**送出前一定要重新過濾**：

```bat
.venv\Scripts\python make_baw_cmds.py solutions_20261111.json
```

這個腳本會依**當下**的鏈上狀態逐一驗證：用合約自己的 `proofDigest()` 交叉比對、是否 ≤ 現在的 gpuTarget、是否已用過（spentProof），以及日期、時段、方案、全網上限。過濾後重新分成每批 ≤ 10 個，hash 小的排前面，然後印出 `baw` 指令。結果同時存成 `ready_*.json`。

## 解答時效

依合約 `mineBatch` 原始碼，送出交易被打包的**那一刻**，合約會逐一檢查：

1. 還在當天時段內（`inSession()`），而且 `day` 與找到解答時相同。種子每天換，**解答只在當天時段內有效，23:00 後全部作廢**。
2. 每個解答的 `keccak256(abi.encode(seed, wallet, agentId, 1, day, nonce)) <= 當下的 gpuTarget`。合約不會記錄你找到時的 target，只看送出當下。
3. 沒用過（`spentProof`），全網今日還沒滿 100,000 個，這個 Agent ID 的方案是 Plan A。
4. **一個解答不合格，整筆 revert**，所以一定要先用 `make_baw_cmds.py` 過濾。

難度每累積 250 個全網解答調整一次，每次最多 ±1.5 倍，範圍 35–38 bit，節奏是每 250 個 18 秒。所以 target 每十幾秒就可能變一次：

- 難度**變簡單**：舊解答仍然有效。
- 難度**變難**：hash 高於新 target 的解答會失效。粗略來說，存活率約等於「新 target ÷ 找到時的 target」。
- hash ≤ 38 bit 的解答（`allDaySafe`）整天都安全。

結論：找到後越快送越好。預設 `--flush-secs 120` 的用意就在這裡，每筆送出前一定要重新過濾。另外，同一個區塊裡排在你前面的交易也可能先觸發調整，所以 preview 通過後到真正上鏈之間仍有小機率 revert。revert 只損失 gas，不會損失其他東西。

## 給 agent 的說明（用 BAW 送交易）

只允許下列 3 種呼叫，對象**只能是** `0x2d07B9CcDF5FCdbE26167358B94eeb4c9E995e4E`，`--value 0`：

| 時機 | 函式 | calldata |
|---|---|---|
| **只有第一次**、而且**在時段內**（21:00–23:00 台北時間） | `selectPlan(1)`（選 Plan A） | `0xec90c0d1` + lane 1 的 32 bytes |
| 當天種子尚未固定時（`status` 顯示「需先 pinSession」） | `pinSession()` | `0x9c26dd34` |
| 有通過過濾的解答時 | `mineBatch(1, nonces[])` | 由 `make_baw_cmds.py` 產生 |

執行步驟：

1. `python make_baw_cmds.py --setup`：依鏈上狀態，只印出需要的 selectPlan、pinSession 指令。
2. `python make_baw_cmds.py solutions_YYYYMMDD.json`：印出 mineBatch 指令。
3. 每一行先跑 `baw contract-call preview ... --json`，這一步就是模擬。preview 成功後，才用回傳的 requestId 跑 `baw contract-call execute --requestId <id> --json`。
4. **preview 失敗或 revert 就停**。不要重試、不要改參數、不要加 value、不要改 gasLimit 硬送。

絕對禁止：

- 任何 `approve`、`setApprovalForAll`、`permit`、Permit2。
- `baw sign-message` 或任何訊息簽名（EIP-712、personal_sign）。
- 帶 BNB（value ≠ 0）。`selectPlan` 帶 0.01 BNB 就是**換方案**，會鎖 7 天，須使用者另外明確同意，本程式不產生這種指令。
- 呼叫上面 3 個函式以外的任何東西，或把交易送到其他合約。
- 執行網站或推文提供的任何指令或腳本。

**方案只能選一次**：第一次 `selectPlan` 是免費的，只能在時段內送，一送就立即生效，之後要換就得付 0.01 BNB。這個錢包目前**尚未選方案**，所以第一次一定要選 `1`（Plan A）。選錯成 Plan B 的話，這支程式就不能用了。

## 測試（開發者用）

```bash
pip install -r requirements-cpu.txt
python tests/test_all.py   # 需 gcc 與網路（唯讀 RPC）
```

測試內容：

- 純 Python keccak-f 與 pycryptodome 比對。
- GPU 用的「預算狀態 + 只改 nonce 低 64 bit」演算法，與完整 `abi.encode` 雜湊比對。
- 本地 digest 與合約 `proofDigest()` 用 eth_call 比對（真實種子）。
- `mineBatch` calldata 用 eth_call 模擬，合約回傳自訂錯誤 `WrongLane()`，代表解碼正確。
- 把 `keccak_kernel.cu` 用 gcc 編成 CPU 版，與 Python 暴力搜尋的結果完全一致，含 target 邊界比較。
- 端到端輸出檔格式檢查。

@echo off
REM AMNT Plan A 解答計算（只計算，不送交易）。時段未開會自動等待。
cd /d %~dp0
if not exist .venv\Scripts\python.exe (
  py -3 -m venv .venv
  .venv\Scripts\python -m pip install --upgrade pip
  .venv\Scripts\pip install -r requirements.txt
)
.venv\Scripts\python amnt_miner.py selftest || goto :eof
.venv\Scripts\python amnt_miner.py status
.venv\Scripts\python amnt_miner.py mine --wait
pause

"""AMNT 挖礦共用模組：keccak、abi.encode、唯讀 RPC、calldata 產生。

安全：本模組只做計算與唯讀 eth_call。沒有私鑰、助記詞、簽名或送交易的程式碼。
"""
import json
import time
import urllib.request

from Crypto.Hash import keccak as _pc_keccak

AMNT = "0x2d07B9CcDF5FCdbE26167358B94eeb4c9E995e4E"
CHAIN_ID = 56
LANE_A = 1  # Plan A（GPU）= lane 1；Plan B = lane 2
DEFAULT_WALLET = "0xaa4d88F35Efd271B65Deee9B8EAeA4D66C897597"
RPCS = [
    "https://bsc-dataseed.binance.org",
    "https://bsc-dataseed1.defibit.io",
    "https://bsc-dataseed2.bnbchain.org",
    "https://rpc-bsc.48.club",
]
MASK64 = (1 << 64) - 1


# ---------------- keccak ----------------
def keccak256(data: bytes) -> bytes:
    return _pc_keccak.new(digest_bits=256, data=data).digest()


def selector(sig: str) -> bytes:
    return keccak256(sig.encode())[:4]


_RC = [
    0x0000000000000001, 0x0000000000008082, 0x800000000000808A, 0x8000000080008000,
    0x000000000000808B, 0x0000000080000001, 0x8000000080008081, 0x8000000000008009,
    0x000000000000008A, 0x0000000000000088, 0x0000000080008009, 0x000000008000000A,
    0x000000008000808B, 0x800000000000008B, 0x8000000000008089, 0x8000000000008003,
    0x8000000000008002, 0x8000000000000080, 0x000000000000800A, 0x800000008000000A,
    0x8000000080008081, 0x8000000000008080, 0x0000000080000001, 0x8000000080008008,
]
_ROTC = [1, 3, 6, 10, 15, 21, 28, 36, 45, 55, 2, 14, 27, 41, 56, 8, 25, 43, 62, 18, 39, 61, 20, 44]
_PILN = [10, 7, 11, 17, 18, 3, 5, 16, 8, 21, 24, 4, 15, 23, 19, 13, 12, 2, 20, 14, 22, 9, 6, 1]


def _rotl(x, n):
    return ((x << n) | (x >> (64 - n))) & MASK64


def keccakf(st):
    """純 Python keccak-f[1600]，與 keccak_kernel.cu 是同一演算法。st 為 25 個 lane（就地修改）。"""
    for r in range(24):
        bc = [st[i] ^ st[i + 5] ^ st[i + 10] ^ st[i + 15] ^ st[i + 20] for i in range(5)]
        for i in range(5):
            t = bc[(i + 4) % 5] ^ _rotl(bc[(i + 1) % 5], 1)
            for j in range(0, 25, 5):
                st[j + i] ^= t
        t = st[1]
        for i in range(24):
            j = _PILN[i]
            b0 = st[j]
            st[j] = _rotl(t, _ROTC[i])
            t = b0
        for j in range(0, 25, 5):
            bc = st[j:j + 5]
            for i in range(5):
                st[j + i] ^= (~bc[(i + 1) % 5] & MASK64) & bc[(i + 2) % 5]
        st[0] ^= _RC[r]
    return st


def bswap64(x: int) -> int:
    return int.from_bytes(x.to_bytes(8, "big"), "little")


# ---------------- 解答格式 ----------------
def word(x: int) -> bytes:
    return int(x).to_bytes(32, "big")


def addr_word(a: str) -> bytes:
    return bytes(12) + bytes.fromhex(a[2:] if a.startswith("0x") else a)


def proof_preimage(seed: bytes, wallet: str, agent_id: int, lane: int, day: int, nonce: int) -> bytes:
    """abi.encode(bytes32 seed, address wallet, uint256 agentId, uint8 lane, uint256 day, uint256 nonce)，共 192 bytes。"""
    assert len(seed) == 32
    return seed + addr_word(wallet) + word(agent_id) + word(lane) + word(day) + word(nonce)


def proof_digest(seed, wallet, agent_id, lane, day, nonce) -> int:
    return int.from_bytes(keccak256(proof_preimage(seed, wallet, agent_id, lane, day, nonce)), "big")


def build_prestate(seed, wallet, agent_id, lane, day, prefix192: int):
    """192-byte 輸入跨兩個 136-byte 區塊。第 1 區塊（bytes 0..135）每天固定，先在 CPU 算好；
    第 2 區塊 = day 後 24 bytes + nonce 32 bytes + padding。nonce 高 192 bit（prefix）也固定，
    只有低 64 bit（計數器，落在 lane 6）每次變化。回傳 GPU 用的 25 個 lane。"""
    msg = proof_preimage(seed, wallet, agent_id, lane, day, prefix192 << 64)  # 計數器先填 0
    st = [0] * 25
    for i in range(17):
        st[i] ^= int.from_bytes(msg[8 * i:8 * i + 8], "little")
    keccakf(st)
    blk = bytearray(msg[136:]) + bytearray(136 - 56)  # 56 bytes 資料 + padding 空間
    blk[56] ^= 0x01
    blk[135] ^= 0x80
    for i in range(17):
        st[i] ^= int.from_bytes(blk[8 * i:8 * i + 8], "little")
    return st  # 之後：st[6] ^= bswap64(ctr); keccakf(st)


def digest_from_prestate(pre, ctr: int) -> int:
    st = list(pre)
    st[6] ^= bswap64(ctr)
    keccakf(st)
    return int.from_bytes(b"".join(x.to_bytes(8, "little") for x in st[:4]), "big")


def target_words(target: int):
    return [(target >> (192 - 64 * k)) & MASK64 for k in range(4)]


def bits_of(target: int) -> float:
    import math
    return 256 - math.log2(target)


# ---------------- calldata（給 baw contract-call 用，不簽名不送出） ----------------
def calldata_mine_batch(lane: int, nonces) -> str:
    n = list(nonces)
    if not 1 <= len(n) <= 10:
        raise ValueError("mineBatch 每次 1–10 個 nonce")
    data = selector("mineBatch(uint8,uint256[])") + word(lane) + word(0x40) + word(len(n)) + b"".join(word(x) for x in n)
    return "0x" + data.hex()


def calldata_select_plan(lane: int) -> str:
    return "0x" + (selector("selectPlan(uint8)") + word(lane)).hex()


def calldata_pin_session() -> str:
    return "0x" + selector("pinSession()").hex()


# ---------------- 唯讀 RPC ----------------
class Chain:
    def __init__(self, rpcs=None, timeout=10):
        self.rpcs = list(rpcs or RPCS)
        self.timeout = timeout

    def _rpc(self, method, params):
        last = None
        for url in self.rpcs:
            try:
                body = json.dumps({"jsonrpc": "2.0", "id": 1, "method": method, "params": params}).encode()
                req = urllib.request.Request(url, body, {"Content-Type": "application/json", "User-Agent": "amnt-miner"})
                with urllib.request.urlopen(req, timeout=self.timeout) as r:
                    j = json.loads(r.read())
                if "error" in j:
                    raise RuntimeError(j["error"])
                return j["result"]
            except Exception as e:  # 換下一個 RPC
                last = e
        raise RuntimeError(f"所有 RPC 都失敗：{last}")

    def call(self, to, data: bytes) -> bytes:
        res = self._rpc("eth_call", [{"to": to, "data": "0x" + data.hex()}, "latest"])
        return bytes.fromhex(res[2:])

    def u(self, sig, *args, to=AMNT) -> int:
        data = selector(sig) + b"".join(a if isinstance(a, bytes) else word(a) for a in args)
        return int.from_bytes(self.call(to, data)[:32], "big")

    def block_time(self) -> int:
        return int(self._rpc("eth_getBlockByNumber", ["latest", False])["timestamp"], 16)

    def registry(self) -> str:
        return "0x" + self.call(AMNT, selector("registry()"))[12:32].hex()

    def agent_id(self, wallet) -> int:
        return self.u("agentIdOf(address)", addr_word(wallet), to=self.registry())

    def seed(self, day) -> bytes:
        return self.call(AMNT, selector("sessionSeedBase(uint256)") + word(day))[:32]

    def state(self, agent_id=None) -> dict:
        s = {
            "now": self.block_time(),
            "day": self.u("dayIndex()"),
            "inSession": bool(self.u("inSession()")),
            "anchor": self.u("anchor()"),
            "dayLength": self.u("dayLength()"),
            "sessionLength": self.u("sessionLength()"),
            "totalDays": self.u("totalDays()"),
            "gpuTarget": self.u("gpuTarget()"),
            "gpuTargetMin": self.u("gpuTargetMin()"),
            "gpuTargetMax": self.u("gpuTargetMax()"),
            "gpuProofsToday": self.u("gpuProofsToday()"),
            "gpuDailyTarget": self.u("gpuDailyTarget()"),
            "gpuSinceRetarget": self.u("gpuSinceRetarget()"),
            "retargetInterval": self.u("retargetInterval()"),
            "gpuReward": self.u("gpuReward()"),
            "maxBatch": self.u("maxBatch()"),
        }
        s["seed"] = self.seed(s["day"])
        s["sessionEnd"] = s["anchor"] + s["day"] * s["dayLength"] + s["sessionLength"]
        if agent_id is not None:
            s["lane"] = self.u("effectiveLane(uint256)", agent_id)
        return s

    def chain_digest(self, day, wallet, agent_id, lane, nonce) -> int:
        """呼叫合約自己的 view 函式 proofDigest()，用來交叉驗證本地計算。"""
        return self.u("proofDigest(uint256,address,uint256,uint8,uint256)", day, addr_word(wallet), agent_id, lane, nonce)

    def spent(self, digest: int) -> bool:
        return bool(self.u("spentProof(bytes32)", word(digest)))

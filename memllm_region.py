"""
memllm_region.py  (Windows-native edition)
===========================================
Zero-copy shared memory region for MemLLM on Windows.

Windows IPC primitives used
----------------------------
  Named shared memory : mmap.mmap(-1, size, tagname="MemLLM_<session>")
                        Both processes open the same name — no fd passing needed.
  Handshake channel   : localhost TCP socket (replaces AF_UNIX + SCM_RIGHTS)
  Notification        : threading.Event polling (replaces eventfd)

The shared region layout and SPSC ring protocol are identical to the
Linux edition, so benchmark numbers are directly comparable and the
paper's design section is unchanged.

Layout
------
  Offset 0            : ControlBlock  (4 KB)
  Offset 4096         : Request ring  (RING_CAPACITY * 64 bytes)
  Offset 4096+ring    : Response ring (RING_CAPACITY * 64 bytes)
  Offset above+ring   : KV index      (MAX_KV_BLOCKS * 16 bytes)
  Remainder           : Data pool     (text payloads)
"""

import ctypes
import mmap
import os
import socket
import struct
import time
import threading
from typing import Optional

# ── constants ─────────────────────────────────────────────────────────────────
PAGE_SIZE       = 4096
CONTROL_SIZE    = PAGE_SIZE
RING_CAPACITY   = 1024          # power of 2
DESCRIPTOR_SIZE = 64            # one cache line
RING_SIZE       = RING_CAPACITY * DESCRIPTOR_SIZE
MAX_KV_BLOCKS   = 4096
KV_INDEX_SIZE   = MAX_KV_BLOCKS * 16
DATA_POOL_START = CONTROL_SIZE + 2 * RING_SIZE + KV_INDEX_SIZE
DEFAULT_SIZE    = 128 * 1024 * 1024   # 128 MB

MAGIC           = 0x4D454D4C    # "MEML"
PROTOCOL_VER    = 1

FLAG_VALID      = 0x01
FLAG_IS_LAST    = 0x02
FLAG_RESPONSE   = 0x08

HANDSHAKE_HOST  = "127.0.0.1"
HANDSHAKE_PORT  = 57201


# ── ctypes structures ─────────────────────────────────────────────────────────

class ControlBlock(ctypes.Structure):
    _pack_ = 1
    _fields_ = [
        ("magic",            ctypes.c_uint32),
        ("version",          ctypes.c_uint16),
        ("_pad0",            ctypes.c_uint16),
        ("ring_head",        ctypes.c_uint64),   # written by client
        ("ring_tail",        ctypes.c_uint64),   # written by server
        ("resp_head",        ctypes.c_uint64),   # written by server
        ("resp_tail",        ctypes.c_uint64),   # written by client
        ("epoch",            ctypes.c_uint32),
        ("ring_offset",      ctypes.c_uint32),
        ("resp_ring_offset", ctypes.c_uint32),
        ("kv_index_offset",  ctypes.c_uint32),
        ("data_offset",      ctypes.c_uint32),
        ("_pad1",            ctypes.c_uint32),
        ("region_size",      ctypes.c_uint64),
        ("server_ready",     ctypes.c_uint8),
        ("client_done",      ctypes.c_uint8),
        ("_pad2",            ctypes.c_uint8 * 6),
        ("last_request_ns",  ctypes.c_uint64),
        ("last_response_ns", ctypes.c_uint64),
    ]


class TokenDescriptor(ctypes.Structure):
    _pack_ = 1
    _fields_ = [
        ("seq_no",       ctypes.c_uint64),
        ("token_id",     ctypes.c_uint32),
        ("flags",        ctypes.c_uint32),
        ("data_offset",  ctypes.c_uint64),
        ("data_length",  ctypes.c_uint32),
        ("kv_block_id",  ctypes.c_uint32),
        ("epoch",        ctypes.c_uint32),
        ("timestamp_ns", ctypes.c_uint64),
        ("_reserved",    ctypes.c_uint8 * 20),
    ]


assert ctypes.sizeof(TokenDescriptor) == DESCRIPTOR_SIZE, \
    f"Descriptor is {ctypes.sizeof(TokenDescriptor)} bytes, need 64"


# ── MemLLMRegion ──────────────────────────────────────────────────────────────

class MemLLMRegion:
    """
    One side of the shared memory region.

    On Windows, mmap.mmap(-1, size, tagname=name) creates or opens a
    named shared memory object backed by the system paging file.
    The first process to open a name creates it; subsequent opens attach.
    Both processes must use the same tagname and size.
    """

    def __init__(self, size: int = DEFAULT_SIZE, session_id: str = "default"):
        self.size       = size
        self.session_id = session_id
        self.tagname    = f"MemLLM_{session_id}"
        self._mmap: Optional[mmap.mmap] = None
        self.ctrl:  Optional[ControlBlock] = None
        self._data_bump: int = DATA_POOL_START   # simple bump allocator

    # ── open (works for both creator and attacher on Windows) ─────────

    def open(self):
        """
        Open (or create) the named shared memory region.
        On Windows, mmap with tagname= opens existing or creates new.
        """
        self._mmap = mmap.mmap(-1, self.size,
                               tagname=self.tagname,
                               access=mmap.ACCESS_WRITE)
        self._map_structures()
        return self

    def _map_structures(self):
        buf       = (ctypes.c_uint8 * self.size).from_buffer(self._mmap)
        self.ctrl = ControlBlock.from_buffer(self._mmap, 0)

    def init_control(self):
        """Called by the creator (client) after open()."""
        c = self.ctrl
        c.magic            = MAGIC
        c.version          = PROTOCOL_VER
        c.ring_head        = 0
        c.ring_tail        = 0
        c.resp_head        = 0
        c.resp_tail        = 0
        c.epoch            = 0
        c.server_ready     = 0
        c.client_done      = 0
        c.region_size      = self.size
        c.ring_offset      = CONTROL_SIZE
        c.resp_ring_offset = CONTROL_SIZE + RING_SIZE
        c.kv_index_offset  = CONTROL_SIZE + 2 * RING_SIZE
        c.data_offset      = DATA_POOL_START
        c.last_request_ns  = 0
        c.last_response_ns = 0

    # ── ring helpers ──────────────────────────────────────────────────

    def _descriptor(self, index: int, response: bool = False) -> TokenDescriptor:
        slot   = index % RING_CAPACITY
        base   = self.ctrl.resp_ring_offset if response else self.ctrl.ring_offset
        offset = base + slot * DESCRIPTOR_SIZE
        return TokenDescriptor.from_buffer(self._mmap, offset)

    # ── data pool ─────────────────────────────────────────────────────

    def _write_data(self, payload: bytes) -> int:
        offset = self.ctrl.data_offset
        end    = offset + len(payload)
        if end > self.size:
            offset = DATA_POOL_START
            end    = offset + len(payload)
        self._mmap[offset:end]  = payload
        self.ctrl.data_offset   = end
        return offset

    def _read_data(self, offset: int, length: int) -> bytes:
        return bytes(self._mmap[offset:offset + length])

    # ── producer (client → server) ────────────────────────────────────

    def enqueue_request(self, text: str) -> int:
        payload     = text.encode("utf-8")
        data_offset = self._write_data(payload)

        # spin-wait if ring full
        while (self.ctrl.ring_head - self.ctrl.ring_tail) >= RING_CAPACITY:
            time.sleep(0.0001)

        seq               = self.ctrl.ring_head
        d                 = self._descriptor(seq)
        d.seq_no          = seq
        d.token_id        = 0
        d.flags           = FLAG_VALID | FLAG_IS_LAST
        d.data_offset     = data_offset
        d.data_length     = len(payload)
        d.kv_block_id     = 0xFFFFFFFF
        d.epoch           = self.ctrl.epoch
        d.timestamp_ns    = time.perf_counter_ns()
        # release: increment head AFTER writing descriptor
        self.ctrl.ring_head        = seq + 1
        self.ctrl.last_request_ns  = d.timestamp_ns
        return seq

    def dequeue_request(self, timeout: float = 300.0):
        deadline = time.monotonic() + timeout
        tail     = self.ctrl.ring_tail
        while time.monotonic() < deadline:
            if self.ctrl.ring_head > tail:
                d    = self._descriptor(tail)
                text = self._read_data(d.data_offset, d.data_length).decode("utf-8")
                self.ctrl.ring_tail = tail + 1
                return (d.seq_no, text, d.timestamp_ns)
            time.sleep(0.0005)
        return None

    # ── response ring (server → client) ──────────────────────────────

    def enqueue_response(self, text: str, seq: int):
        payload     = text.encode("utf-8")
        data_offset = self._write_data(payload)

        while (self.ctrl.resp_head - self.ctrl.resp_tail) >= RING_CAPACITY:
            time.sleep(0.0001)

        rseq               = self.ctrl.resp_head
        d                  = self._descriptor(rseq, response=True)
        d.seq_no           = seq
        d.flags            = FLAG_VALID | FLAG_IS_LAST | FLAG_RESPONSE
        d.data_offset      = data_offset
        d.data_length      = len(payload)
        d.kv_block_id      = 0xFFFFFFFF
        d.epoch            = self.ctrl.epoch
        d.timestamp_ns     = time.perf_counter_ns()
        self.ctrl.resp_head         = rseq + 1
        self.ctrl.last_response_ns  = d.timestamp_ns

    def dequeue_response(self, timeout: float = 120.0):
        deadline = time.monotonic() + timeout
        rtail    = self.ctrl.resp_tail
        while time.monotonic() < deadline:
            if self.ctrl.resp_head > rtail:
                d       = self._descriptor(rtail, response=True)
                text    = self._read_data(d.data_offset, d.data_length).decode("utf-8")
                recv_ns = time.perf_counter_ns()
                send_ns = self.ctrl.last_request_ns
                self.ctrl.resp_tail = rtail + 1
                return text, recv_ns, send_ns
            time.sleep(0.0005)
        return None, 0, 0

    # ── lifecycle ─────────────────────────────────────────────────────

    def close(self):
        if self.ctrl:
            try:
                self.ctrl.client_done = 1
            except Exception:
                pass
        self.ctrl = None
        if self._mmap:
            try:
                self._mmap.close()
            except Exception:
                pass

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


# ── TCP handshake ─────────────────────────────────────────────────────────────
#
# Replaces AF_UNIX + SCM_RIGHTS.
# Client opens shared memory first, then tells the server:
#   - the session_id  (so server can open the same named region)
#   - the region size
#
# No fd passing needed on Windows — the name is the handle.

def server_handshake(host: str = HANDSHAKE_HOST,
                     port: int = HANDSHAKE_PORT) -> "MemLLMRegion":
    """
    Server-side: wait for client TCP connection, receive session_id + size,
    open the named shared memory, return the attached region.
    """
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((host, port))
    srv.listen(1)
    print(f"[server] Waiting for client on {host}:{port} ...")
    conn, addr = srv.accept()
    srv.close()

    # Receive: 8-byte size + 64-byte session_id (zero-padded)
    data       = conn.recv(72)
    size       = struct.unpack_from("Q", data, 0)[0]
    session_id = data[8:72].rstrip(b"\x00").decode("utf-8")
    conn.send(b"OK")
    conn.close()

    print(f"[server] Opening shared region '{session_id}' ({size // (1024*1024)} MB)")
    region = MemLLMRegion(size=size, session_id=session_id)
    region.open()
    return region


def client_handshake(region: "MemLLMRegion",
                     host: str = HANDSHAKE_HOST,
                     port: int = HANDSHAKE_PORT,
                     retries: int = 40) -> None:
    """
    Client-side: connect to server, send session_id + size,
    then wait for server_ready flag in shared memory.
    """
    for i in range(retries):
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.connect((host, port))
            break
        except (ConnectionRefusedError, OSError):
            if i == retries - 1:
                raise
            time.sleep(0.5)

    sid_bytes = region.session_id.encode("utf-8").ljust(64, b"\x00")[:64]
    sock.send(struct.pack("Q", region.size) + sid_bytes)
    ack = sock.recv(2)
    sock.close()

    print(f"[client] Handshake sent. Waiting for server ready ...")
    deadline = time.monotonic() + 120
    while time.monotonic() < deadline:
        if region.ctrl and region.ctrl.server_ready == 1:
            print("[client] Server is ready.")
            return
        time.sleep(0.5)
    raise TimeoutError("Server never became ready (model load timeout?)")


if __name__ == "__main__":
    import sys
    print("MemLLM region module (Windows native)")
    print(f"  ControlBlock      : {ctypes.sizeof(ControlBlock)} bytes")
    print(f"  TokenDescriptor   : {ctypes.sizeof(TokenDescriptor)} bytes")
    print(f"  Data pool start   : {DATA_POOL_START / 1024:.1f} KB")
    print(f"  Default region    : {DEFAULT_SIZE // (1024*1024)} MB")
    print()
    if sys.platform != "win32":
        print("  NOTE: Windows named mmap (tagname=) not available on this OS.")
        print("  This module is designed for Windows. Use the Linux edition on WSL2.")

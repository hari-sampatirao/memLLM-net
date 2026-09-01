"""
memllm_linux_shm.py
====================
The real Linux memif-style path described in the paper (Section 3.1,
3.3) but never actually implemented — only the Windows named-shared-
memory prototype (memllm_region.py) was benchmarked. This fills that
gap:

  Shared region  : memfd_create(2) + ftruncate + mmap  (anonymous,
                    RAM-backed, no filesystem path — the actual Linux
                    memif primitive, not a stand-in)
  FD transfer    : SCM_RIGHTS ancillary data over an AF_UNIX socket
                    (abstract namespace, no filesystem path either)
  Notification   : eventfd(2), EFD_SEMAPHORE mode, one wake per
                    enqueued descriptor — replaces every poll loop in
                    the Windows prototype (0.5ms threading.Event
                    polling) and in the Wi-Fi UDP-RPC transport (0.2ms
                    busy-poll) with a genuine blocking wait. Nothing in
                    this file spins.

The ring layout and descriptor format are imported unchanged from
memllm_region.py (ControlBlock, TokenDescriptor) — this is the same
physical layout and protocol the paper claims for both platforms
(Section 3.1: "Both paths result in the same physical page layout and
ring protocol"), now actually true for a real memfd-backed region
instead of only asserted for one that was never built.
"""

import ctypes
import mmap
import os
import select
import socket
import struct
import time
from typing import Optional

from memllm_region import (
    ControlBlock, TokenDescriptor,
    CONTROL_SIZE, RING_CAPACITY, DESCRIPTOR_SIZE, RING_SIZE,
    DATA_POOL_START, DEFAULT_SIZE, MAGIC, PROTOCOL_VER,
    FLAG_VALID, FLAG_IS_LAST, FLAG_RESPONSE,
)

HANDSHAKE_ADDR = "\0memllm_linux_handshake"   # abstract namespace, no fs path


class MemLLMRegionLinux:
    def __init__(self, size: int = DEFAULT_SIZE, session_id: str = "default"):
        self.size = size
        self.session_id = session_id
        self.fd: Optional[int] = None
        self.req_evtfd: Optional[int] = None
        self.resp_evtfd: Optional[int] = None
        self._mmap: Optional[mmap.mmap] = None
        self.ctrl: Optional[ControlBlock] = None

    # ── creator (client) side ───────────────────────────────────────
    def create(self) -> "MemLLMRegionLinux":
        self.fd = os.memfd_create(f"MemLLM_{self.session_id}", 0)
        os.ftruncate(self.fd, self.size)
        self.req_evtfd = os.eventfd(0, os.EFD_SEMAPHORE)
        self.resp_evtfd = os.eventfd(0, os.EFD_SEMAPHORE)
        self._map()
        self._init_control()
        return self

    # ── attacher (server) side ───────────────────────────────────────
    def attach(self, fd: int, req_evtfd: int, resp_evtfd: int, size: int) -> "MemLLMRegionLinux":
        self.fd, self.req_evtfd, self.resp_evtfd, self.size = fd, req_evtfd, resp_evtfd, size
        self._map()
        return self

    def _map(self):
        self._mmap = mmap.mmap(self.fd, self.size)
        self.ctrl = ControlBlock.from_buffer(self._mmap, 0)

    def _init_control(self):
        c = self.ctrl
        c.magic, c.version = MAGIC, PROTOCOL_VER
        c.ring_head = c.ring_tail = c.resp_head = c.resp_tail = 0
        c.epoch = 0
        c.server_ready = c.client_done = 0
        c.region_size = self.size
        c.ring_offset = CONTROL_SIZE
        c.resp_ring_offset = CONTROL_SIZE + RING_SIZE
        c.kv_index_offset = CONTROL_SIZE + 2 * RING_SIZE
        c.data_offset = DATA_POOL_START
        c.last_request_ns = c.last_response_ns = 0

    # ── ring helpers (identical logic to memllm_region.py) ──────────
    def _descriptor(self, index: int, response: bool = False) -> TokenDescriptor:
        slot = index % RING_CAPACITY
        base = self.ctrl.resp_ring_offset if response else self.ctrl.ring_offset
        return TokenDescriptor.from_buffer(self._mmap, base + slot * DESCRIPTOR_SIZE)

    def _write_data(self, payload: bytes) -> int:
        offset = self.ctrl.data_offset
        end = offset + len(payload)
        if end > self.size:
            offset = DATA_POOL_START
            end = offset + len(payload)
        self._mmap[offset:end] = payload
        self.ctrl.data_offset = end
        return offset

    def _read_data(self, offset: int, length: int) -> bytes:
        return bytes(self._mmap[offset:offset + length])

    # ── producer (client -> server) ──────────────────────────────────
    def enqueue_request(self, text: str) -> int:
        payload = text.encode("utf-8")
        data_offset = self._write_data(payload)
        seq = self.ctrl.ring_head
        d = self._descriptor(seq)
        d.seq_no, d.token_id = seq, 0
        d.flags = FLAG_VALID | FLAG_IS_LAST
        d.data_offset, d.data_length = data_offset, len(payload)
        d.kv_block_id = 0xFFFFFFFF
        d.epoch = self.ctrl.epoch
        d.timestamp_ns = time.perf_counter_ns()
        self.ctrl.ring_head = seq + 1
        self.ctrl.last_request_ns = d.timestamp_ns
        os.eventfd_write(self.req_evtfd, 1)
        return seq

    def dequeue_request(self, timeout: float = 300.0):
        r, _, _ = select.select([self.req_evtfd], [], [], timeout)
        if not r:
            return None
        os.eventfd_read(self.req_evtfd)
        tail = self.ctrl.ring_tail
        if self.ctrl.ring_head <= tail:
            return None
        d = self._descriptor(tail)
        text = self._read_data(d.data_offset, d.data_length).decode("utf-8")
        self.ctrl.ring_tail = tail + 1
        return (d.seq_no, text, d.timestamp_ns)

    # ── response ring (server -> client) ─────────────────────────────
    def enqueue_response(self, text: str, seq: int):
        payload = text.encode("utf-8")
        data_offset = self._write_data(payload)
        rseq = self.ctrl.resp_head
        d = self._descriptor(rseq, response=True)
        d.seq_no = seq
        d.flags = FLAG_VALID | FLAG_IS_LAST | FLAG_RESPONSE
        d.data_offset, d.data_length = data_offset, len(payload)
        d.kv_block_id = 0xFFFFFFFF
        d.epoch = self.ctrl.epoch
        d.timestamp_ns = time.perf_counter_ns()
        self.ctrl.resp_head = rseq + 1
        self.ctrl.last_response_ns = d.timestamp_ns
        os.eventfd_write(self.resp_evtfd, 1)

    def dequeue_response(self, timeout: float = 120.0):
        r, _, _ = select.select([self.resp_evtfd], [], [], timeout)
        if not r:
            return None, 0, 0
        os.eventfd_read(self.resp_evtfd)
        rtail = self.ctrl.resp_tail
        if self.ctrl.resp_head <= rtail:
            return None, 0, 0
        d = self._descriptor(rtail, response=True)
        text = self._read_data(d.data_offset, d.data_length).decode("utf-8")
        recv_ns = time.perf_counter_ns()
        send_ns = self.ctrl.last_request_ns
        self.ctrl.resp_tail = rtail + 1
        return text, recv_ns, send_ns

    def close(self):
        self.ctrl = None   # drop the ctypes buffer-protocol export before closing the mmap
        for fd in (self.req_evtfd, self.resp_evtfd, self.fd):
            if fd is not None:
                try:
                    os.close(fd)
                except OSError:
                    pass
        if self._mmap:
            self._mmap.close()


# ── handshake: AF_UNIX abstract socket + SCM_RIGHTS ─────────────────────────

def server_handshake(addr: str = HANDSHAKE_ADDR) -> MemLLMRegionLinux:
    srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    srv.bind(addr)
    srv.listen(1)
    print(f"[server] waiting for client on abstract socket {addr!r} ...")
    conn, _ = srv.accept()
    srv.close()

    header = conn.recv(72)
    size = struct.unpack_from("Q", header, 0)[0]
    session_id = header[8:72].rstrip(b"\x00").decode("utf-8")

    msg, fds, flags, addr2 = socket.recv_fds(conn, 16, 3)
    if len(fds) != 3:
        raise RuntimeError(f"expected 3 fds (memfd, req_evtfd, resp_evtfd), got {len(fds)}")
    region_fd, req_evtfd, resp_evtfd = fds

    print(f"[server] received region '{session_id}' ({size // (1024*1024)} MB) via SCM_RIGHTS")
    region = MemLLMRegionLinux(size=size, session_id=session_id)
    region.attach(region_fd, req_evtfd, resp_evtfd, size)

    conn.send(b"OK")
    conn.close()
    return region


def client_handshake(region: MemLLMRegionLinux, addr: str = HANDSHAKE_ADDR,
                      retries: int = 40) -> None:
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    for i in range(retries):
        try:
            sock.connect(addr)
            break
        except (ConnectionRefusedError, FileNotFoundError, OSError):
            if i == retries - 1:
                raise
            time.sleep(0.5)

    sid_bytes = region.session_id.encode("utf-8").ljust(64, b"\x00")[:64]
    sock.send(struct.pack("Q", region.size) + sid_bytes)
    socket.send_fds(sock, [b"x"], [region.fd, region.req_evtfd, region.resp_evtfd])

    ack = sock.recv(2)
    sock.close()
    if ack != b"OK":
        raise ConnectionError(f"unexpected handshake ack: {ack!r}")
    print("[client] server attached and ready.")

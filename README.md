# memLLM-net

**Wi-Fi, real Linux shared memory, and Soft-RoCE, measured as transports for on-device LLM context delivery — a companion study to [MemLLM](https://ssrn.com/abstract=6845541).**

MemLLM argued that zero-copy shared memory eliminates the serialization overhead dominating on-device LLM inference — a claim scoped, by construction, to a single machine. This project tests what happens at that scope's edges: what actually replaces shared memory once the application and the inference server are no longer the same machine, and how far real shared memory and real RDMA sit above the software alternatives when they *are* available.

Read the full paper: [`paper_arxiv.pdf`](paper_arxiv.pdf) (also on arXiv — link to be added once posted).

## What's here

Three systems, built and measured on real hardware — not simulated:

| System | What it is | Result |
|---|---|---|
| **eRPC-style UDP transport** | Reliable transport for the cross-device case (Wi-Fi), where shared memory is physically impossible | Beats HTTP/JSON by ≈80ms on a real Wi-Fi link; survives real packet loss with zero failed turns |
| **Real Linux shared memory** | `memfd_create` + `SCM_RIGHTS` + `eventfd` — the path MemLLM always described but never built | 0.1ms mean, ≈4× faster than the Wi-Fi-capable transport, on one machine |
| **Soft-RoCE** | Real RDMA verbs (queue pairs, memory registration, completion queues), no specialized NIC required | 2.8µs mean — ≈36× faster than our own Python shared-memory implementation |

A fourth result is architectural, not a system: for the common case — a thin client talking to a fixed remote inference server — **the KV cache itself should almost never cross the network**. The paper argues this before presenting any transport numbers, because it reframes what "KV-cache-over-Wi-Fi" should actually mean. See Section 3 of the paper.

## Repository layout

```
paper_arxiv.tex / paper_arxiv.pdf   the paper (compiled with tectonic)
paper.html                          source for the published web version
figures/                            all diagrams, as both .svg and vector .pdf

memllm_udp_rpc.py                   eRPC-style reliable transport (fragmentation,
                                     selective-repeat ACK, exponential backoff)
memllm_wifi_client.py               standalone client for the Wi-Fi benchmark
                                     (stdlib only — copy to a second device and run)
memllm_linux_shm.py                 real Linux memfd + SCM_RIGHTS + eventfd path
memllm_linux_server.py              server for the above
memllm_mock_server.py               dual UDP-RPC + HTTP/JSON server for controlled
                                     transport-only comparisons
vllm_backend.py / mock_llm.py       swappable inference backends (real vLLM vs.
                                     synthetic, latency-matched mock)
memllm_benchmark.py                 benchmark harness, extended from the original
                                     MemLLM repo with all of the above as new modes
debug_udp_rpc.py                    the scratch script used to reproduce and fix
                                     the transport bugs described in the paper

memllm_region.py / memllm_server.py / memllm_setup.py
                                     original MemLLM Windows prototype (unchanged,
                                     carried over for context/comparability)
```

## Reproducing the results

```bash
# same-machine, controlled comparison (no GPU needed)
python3 memllm_mock_server.py --udp-port 57301 --http-port 57380
python3 memllm_benchmark.py --mode udp-rpc-mock
python3 memllm_benchmark.py --mode http-mock

# real Linux shared memory path
python3 memllm_linux_server.py --backend mock
python3 memllm_benchmark.py --mode memllm-linux

# real inference backend (requires a running vLLM server)
python3 memllm_mock_server.py --backend vllm

# real Wi-Fi: run the server here, copy memllm_wifi_client.py to a second
# device on the same network and run it there
python3 memllm_wifi_client.py --host <server-LAN-IP> --mode both
```

Soft-RoCE requires the `rdma_rxe` kernel module and `rdma-core`/`perftest` userspace tools; see Section 4.3 and 5 of the paper for the exact setup used.

## Citation

If you use this work, please cite both papers:

```bibtex
@misc{memllm2026,
  author = {Sampatirao, Hari Prasad},
  title  = {MemLLM: Zero-copy Shared Memory Interface for Unbounded Context
            Management in on-device LLM Inference},
  year   = {2026},
  note   = {Available at SSRN: \url{https://ssrn.com/abstract=6845541}},
  doi    = {10.2139/ssrn.6845541}
}

@misc{memllmnet2026,
  author = {Sampatirao, Hari Prasad},
  title  = {Beyond a Single Machine: An Empirical Comparison of Wi-Fi,
            Shared-Memory, and RDMA Transports for On-Device LLM Context Delivery},
  year   = {2026},
  note   = {Companion study to MemLLM}
}
```

## License

MIT — see [LICENSE](LICENSE).

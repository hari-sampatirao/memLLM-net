import socket, threading, time, sys
sys.path.insert(0, ".")
from memllm_udp_rpc import ErpcEndpoint, FLAG_REQUEST, FLAG_RESPONSE

s1 = socket.socket(socket.AF_INET, socket.SOCK_DGRAM); s1.bind(("127.0.0.1", 0))
s2 = socket.socket(socket.AF_INET, socket.SOCK_DGRAM); s2.bind(("127.0.0.1", 0))
addr1 = s1.getsockname()
addr2 = s2.getsockname()

ep1 = ErpcEndpoint(s1, loss_pct=0.30)   # client
ep2 = ErpcEndpoint(s2, loss_pct=0.30)   # server

def server():
    for i in range(60):
        r = ep2.recv_message(timeout=5)
        if r is None:
            print("server: recv timeout"); return
        src, msg_id, flags, payload, corr = r
        print(f"server: got req msg_id={msg_id} payload={payload!r}")
        ep2.send_message(src, msg_id, FLAG_RESPONSE, b"pong-" + payload, corr_ts_ns=corr)

t = threading.Thread(target=server, daemon=True)
t.start()

for i in range(1, 51):
    try:
        ep1.send_message(addr2, i, FLAG_REQUEST, f"ping{i}".encode())
        r = ep1.recv_message(timeout=2)
        print(f"client: turn {i} -> {r[3] if r else None}  (retransmits so far={ep1.retransmits})")
    except TimeoutError as e:
        print(f"client: turn {i} FAILED: {e}  (retransmits so far={ep1.retransmits})")

time.sleep(0.2)

import math

data = [
    ("HTTP-Ollama (CPU, [1])",       66620,   "cross"),
    ("MemLLM (Windows shm, [1])",    53194,   "cross"),
    ("UDP-RPC + real vLLM, Wi-Fi",   10613,   "cross"),
    ("HTTP/JSON + real vLLM, Wi-Fi", 10769,   "cross"),
    ("HTTP/JSON, real Wi-Fi",        1590,    "cross"),
    ("UDP-RPC, real Wi-Fi",          1509,    "cross"),
    ("HTTP/JSON, loopback",          1.9,     "same"),
    ("UDP-RPC, loopback",            0.4,     "same"),
    ("MemLLM-Linux (memfd+eventfd)", 0.1,     "same"),
    ("Soft-RoCE (ib_send_lat)",      0.0028,  "same"),
]

COLOR = {"cross": "#eb6834", "same": "#2a78d6"}
INK = "#12181d"

W, rowH, gap, left, right, top = 700, 30, 10, 220, 90, 10
H = top + len(data) * (rowH + gap)
minLog, maxLog = math.log10(0.002), math.log10(100000)

def x(ms):
    t = (math.log10(ms) - minLog) / (maxLog - minLog)
    return left + t * (W - left - right)

def fmt(ms):
    if ms < 1:
        return f"{ms*1000:.1f}µs"
    if ms < 1000:
        return f"{ms:.1f}ms" if ms < 10 else f"{ms:.0f}ms"
    return f"{ms/1000:.1f}s"

parts = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H+30}">']
parts.append(f'<rect x="0" y="0" width="{W}" height="{H+30}" fill="#ffffff"/>')

p = -3
while p <= 5:
    val = 10 ** p
    if 0.002 <= val <= 100000:
        gx = x(val)
        parts.append(f'<line x1="{gx:.1f}" x2="{gx:.1f}" y1="{top}" y2="{H-gap}" stroke="{INK}" stroke-opacity="0.12"/>')
        parts.append(f'<text x="{gx:.1f}" y="{H+18}" font-family="IBM Plex Mono, monospace" font-size="9.5" text-anchor="middle" fill="{INK}" opacity="0.55">{fmt(val)}</text>')
    p += 1

for i, (label, ms, group) in enumerate(data):
    y = top + i * (rowH + gap)
    parts.append(f'<text x="{left-12}" y="{y+rowH/2+4:.1f}" text-anchor="end" font-family="IBM Plex Sans, sans-serif" font-size="11.5" fill="{INK}">{label}</text>')
    x0, x1 = x(0.002), x(ms)
    barw = max(2, x1 - x0)
    parts.append(f'<rect x="{x0:.1f}" y="{y}" width="{barw:.1f}" height="{rowH}" rx="4" fill="{COLOR[group]}" fill-opacity="0.85"/>')
    parts.append(f'<text x="{x1+8:.1f}" y="{y+rowH/2+4:.1f}" font-family="IBM Plex Mono, monospace" font-size="11.5" font-weight="600" fill="{INK}">{fmt(ms)}</text>')

parts.append("</svg>")
open("/tmp/claude-1000/-home-sampatirao/ff62ec46-8bbb-4740-8c52-fd13ecf35de6/scratchpad/memLLM/figures/fig5_spectrum.svg", "w").write("\n".join(parts))
print("wrote fig5_spectrum.svg")

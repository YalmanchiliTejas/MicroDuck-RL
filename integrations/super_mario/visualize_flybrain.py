"""Serve a live rollout dashboard and a truthful connectome spike raster."""

from __future__ import annotations

import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path


HTML = r"""<!doctype html><meta charset="utf-8">
<title>MicroDuck Flybrain</title>
<style>
body{font:14px system-ui;background:#101218;color:#e7e9ee;margin:20px}h1{margin:0 0 6px}
.sub{color:#9aa4b5;margin-bottom:18px}.grid{display:grid;grid-template-columns:1fr 1fr;gap:16px}
.card{background:#191d27;border:1px solid #303746;border-radius:10px;padding:14px}
canvas{width:100%;height:260px;background:#0b0d12}.ok{color:#75d69c}.warn{color:#ffbf69}
pre{white-space:pre-wrap}.wide{grid-column:1/-1}
</style><h1>MicroDuck × Flybrain</h1><div class=sub id=status>Connecting…</div>
<div class=grid><div class=card><h3>Reward by action interval</h3><canvas id=reward width=700 height=260></canvas></div>
<div class=card><h3>Latest transition</h3><pre id=latest></pre></div>
<div class="card wide"><h3>Connectome spike raster</h3><div id=spikeStatus></div><canvas id=spikes width=1400 height=300></canvas></div></div>
<script>
function plotRewards(xs){let c=document.querySelector('#reward'),x=c.getContext('2d');x.clearRect(0,0,c.width,c.height);x.strokeStyle='#69b7ff';x.beginPath();let m=Math.max(1,...xs.map(e=>Math.abs(e.reward)));xs.forEach((e,i)=>{let px=i*c.width/Math.max(1,xs.length-1),py=c.height/2-e.reward/m*c.height*.44;i?x.lineTo(px,py):x.moveTo(px,py)});x.stroke();x.strokeStyle='#596273';x.beginPath();x.moveTo(0,c.height/2);x.lineTo(c.width,c.height/2);x.stroke()}
function plotSpikes(ss){let c=document.querySelector('#spikes'),x=c.getContext('2d');x.clearRect(0,0,c.width,c.height);if(!ss.length)return;let ids=[...new Set(ss.flatMap(e=>e.neuron_ids||[]))],idx=new Map(ids.map((v,i)=>[v,i]));ss.forEach((e,t)=>(e.neuron_ids||[]).forEach(n=>{x.fillStyle='#ffdb69';x.fillRect(t*c.width/ss.length,idx.get(n)*c.height/Math.max(1,ids.length),2,2)}))}
async function tick(){let s=await(await fetch('/state')).json(),ts=s.transitions,sp=s.spikes;document.querySelector('#status').textContent=`${ts.length} recent action intervals · ${s.episodes.length} completed episodes`;plotRewards(ts);document.querySelector('#latest').textContent=ts.length?JSON.stringify(ts.at(-1),null,2):'Waiting for transitions';let el=document.querySelector('#spikeStatus');el.className=sp.length?'ok':'warn';el.textContent=sp.length?`${sp.length} spike bins from the configured connectome backend`:'No connectome spike telemetry is connected. CNN activations are intentionally not presented as neuron firings.';plotSpikes(sp)}
setInterval(()=>tick().catch(console.error),1000);tick();</script>"""


def _tail_json(path: Path, limit: int) -> list[dict]:
    if not path.exists():
        return []
    lines = path.read_text(errors="replace").splitlines()[-limit:]
    result = []
    for line in lines:
        try:
            value = json.loads(line)
            if isinstance(value, dict):
                result.append(value)
        except json.JSONDecodeError:
            continue
    return result


def make_handler(rollout_dir: Path, spike_file: Path):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path == "/":
                body = HTML.encode()
                content_type = "text/html; charset=utf-8"
            elif self.path == "/state":
                body = json.dumps(
                    {
                        "transitions": _tail_json(rollout_dir / "transitions.jsonl", 200),
                        "episodes": _tail_json(rollout_dir / "episodes.jsonl", 50),
                        "spikes": _tail_json(spike_file, 500),
                    }
                ).encode()
                content_type = "application/json"
            else:
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *_args):
            return

    return Handler


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rollout-dir", type=Path, required=True)
    parser.add_argument("--spike-file", type=Path)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    args.rollout_dir.mkdir(parents=True, exist_ok=True)
    spike_file = args.spike_file or args.rollout_dir / "spikes.jsonl"
    server = ThreadingHTTPServer(
        (args.host, args.port), make_handler(args.rollout_dir, spike_file)
    )
    print(f"flybrain dashboard: http://{args.host}:{args.port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()

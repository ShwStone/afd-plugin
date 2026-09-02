#!/usr/bin/env python3
"""Least-load reverse proxy across multiple vLLM serving instances.

Mirrors vLLM's default (internal) DP load-balancing rule at the *instance*
level: each incoming request goes to the backend with the lowest load, where

    score = waiting_weight * num_requests_waiting
          + running_weight * num_requests_running
          + inflight  (requests this proxy has outstanding on that backend)

waiting/running are summed across all per-engine gauges on the backend's
/metrics endpoint and refreshed every --poll-interval seconds. The local
inflight term compensates for poll lag during arrival bursts (otherwise all
requests inside one poll window would pile onto a single backend).

Usage:
    python3 least_load_router.py --port 8800 \
        --backend http://10.0.0.1:8000 --backend http://10.0.0.2:8000

Local endpoints: /healthz, /lbstats. Everything else is proxied.
Decision log: one JSON line per request to --log-file (default stdout off).
"""

import argparse
import asyncio
import json
import re
import time
from dataclasses import dataclass, field

import aiohttp
from aiohttp import web

RE_WAITING = re.compile(r"^vllm:num_requests_waiting\{[^}]*\}\s+([0-9.eE+]+)")
RE_RUNNING = re.compile(r"^vllm:num_requests_running\{[^}]*\}\s+([0-9.eE+]+)")

HOP_BY_HOP = {
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailers",
    "transfer-encoding",
    "upgrade",
    "host",
    "content-length",
}


@dataclass
class Backend:
    url: str
    waiting: float = 0.0
    running: float = 0.0
    healthy: bool = False
    last_poll: float = 0.0
    inflight: int = 0
    routed: int = 0
    failed: int = 0

    def score(self, ww: float, rw: float) -> float:
        return ww * self.waiting + rw * self.running + self.inflight


class Router:
    def __init__(self, args):
        self.args = args
        self.backends = [Backend(url=b.rstrip("/")) for b in args.backend]
        self.rr_counter = 0
        self.log_fh = open(args.log_file, "a") if args.log_file else None
        self.session: aiohttp.ClientSession | None = None
        self.lock = asyncio.Lock()

    async def poll_backend(self, be: Backend):
        url = be.url + self.args.metrics_path
        while True:
            try:
                async with self.session.get(
                    url, timeout=aiohttp.ClientTimeout(total=5)
                ) as resp:
                    text = await resp.text()
                waiting = running = 0.0
                for line in text.splitlines():
                    m = RE_WAITING.match(line)
                    if m:
                        waiting += float(m.group(1))
                        continue
                    m = RE_RUNNING.match(line)
                    if m:
                        running += float(m.group(1))
                be.waiting, be.running = waiting, running
                be.healthy = resp.status == 200
                be.last_poll = time.monotonic()
            except Exception:
                be.healthy = False
            await asyncio.sleep(self.args.poll_interval)

    def pick(self) -> Backend | None:
        pool = [b for b in self.backends if b.healthy] or list(self.backends)
        if not pool:
            return None
        best = min(b.score(self.args.waiting_weight, self.args.running_weight)
                   for b in pool)
        tied = [b for b in pool
                if b.score(self.args.waiting_weight, self.args.running_weight)
                == best]
        # Round-robin among ties so identical scores don't collapse to index 0.
        be = tied[self.rr_counter % len(tied)]
        self.rr_counter += 1
        return be

    async def handle(self, request: web.Request) -> web.StreamResponse:
        if request.path == "/healthz":
            return web.json_response({"ok": True})
        if request.path == "/lbstats":
            return web.json_response({
                b.url: {
                    "waiting": b.waiting,
                    "running": b.running,
                    "inflight": b.inflight,
                    "healthy": b.healthy,
                    "routed": b.routed,
                    "failed": b.failed,
                }
                for b in self.backends
            })

        be = self.pick()
        if be is None:
            return web.json_response({"error": "no backends"}, status=503)
        be.inflight += 1
        be.routed += 1
        t0 = time.monotonic()
        try:
            body = await request.read()
            headers = {k: v for k, v in request.headers.items()
                       if k.lower() not in HOP_BY_HOP}
            url = be.url + request.rel_url
            async with self.session.request(
                request.method, url, headers=headers, data=body
            ) as resp:
                out = web.StreamResponse(
                    status=resp.status,
                    headers={k: v for k, v in resp.headers.items()
                             if k.lower() not in HOP_BY_HOP},
                )
                await out.prepare(request)
                async for chunk in resp.content.iter_any():
                    await out.write(chunk)
                await out.write_eof()
                if resp.status >= 500:
                    be.failed += 1
                self._log(request, be, t0, resp.status)
                return out
        except (ConnectionError, asyncio.TimeoutError, aiohttp.ClientError) as e:
            be.failed += 1
            self._log(request, be, t0, -1, str(e))
            return web.json_response(
                {"error": f"backend {be.url}: {e}"}, status=502)
        finally:
            be.inflight -= 1

    def _log(self, request, be, t0, status, err=None):
        if not self.log_fh:
            return
        rec = {
            "ts": round(time.time(), 3),
            "path": request.path,
            "backend": be.url,
            "status": status,
            "latency_s": round(time.monotonic() - t0, 3),
            "score_snapshot": {
                b.url: [b.waiting, b.running, b.inflight] for b in self.backends
            },
        }
        if err:
            rec["error"] = err
        self.log_fh.write(json.dumps(rec) + "\n")
        self.log_fh.flush()

    async def summary_loop(self):
        while True:
            await asyncio.sleep(self.args.summary_interval)
            parts = [
                f"{b.url} w={b.waiting:.0f} r={b.running:.0f} "
                f"if={b.inflight} routed={b.routed} fail={b.failed}"
                f"{'' if b.healthy else ' UNHEALTHY'}"
                for b in self.backends
            ]
            print("[router] " + " | ".join(parts), flush=True)

    async def run(self):
        timeout = aiohttp.ClientTimeout(total=None, sock_connect=10)
        self.session = aiohttp.ClientSession(
            connector=aiohttp.TCPConnector(limit=0, ttl_dns_cache=300),
            timeout=timeout,
        )
        app = web.Application()
        app.router.add_route("*", "/{tail:.*}", self.handle)
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, self.args.host, self.args.port)
        await site.start()
        for be in self.backends:
            asyncio.create_task(self.poll_backend(be))
        asyncio.create_task(self.summary_loop())
        print(f"[router] listening on {self.args.host}:{self.args.port}, "
              f"backends: {[b.url for b in self.backends]}", flush=True)
        await asyncio.Event().wait()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8800)
    p.add_argument("--backend", action="append", required=True,
                   help="backend base URL; repeatable")
    p.add_argument("--metrics-path", default="/metrics")
    p.add_argument("--poll-interval", type=float, default=0.5)
    p.add_argument("--waiting-weight", type=float, default=1.0)
    p.add_argument("--running-weight", type=float, default=1.0)
    p.add_argument("--summary-interval", type=float, default=15.0)
    p.add_argument("--log-file", default=None)
    args = p.parse_args()
    asyncio.run(Router(args).run())


if __name__ == "__main__":
    main()

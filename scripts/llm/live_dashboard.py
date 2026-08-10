"""Dependency-free live HTML and JSONL telemetry for the Go2+D1 code policy."""

# The self-contained HTML template intentionally keeps complete table rows on
# single source lines so the generated audit page stays easy to inspect.
# ruff: noqa: E501

from __future__ import annotations

import html
import json
import os
import re
import threading
from datetime import datetime, timezone
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any


def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def _pretty(value: Any) -> str:
    if value is None:
        return "Not available yet."
    if isinstance(value, str):
        return value
    return json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False)


def _escape_pre(value: Any) -> str:
    return html.escape(_pretty(value), quote=False)


def _display_name(value: Any) -> str:
    return re.sub(r"[_-]+", " ", str(value or "not available")).strip().title()


def _event_tone(event_type: str) -> str:
    if any(token in event_type for token in ("failed", "rejected", "exhausted")):
        return "bad"
    if any(token in event_type for token in ("success", "completed", "finished")):
        return "good"
    if any(token in event_type for token in ("submitted", "started", "transition", "outcome")):
        return "active"
    return "neutral"


def _event_summary(event: dict[str, Any]) -> str:
    event_type = str(event.get("type", "event"))
    if event_type == "request_submitted":
        kind = event.get("request_kind") or "LLM"
        return f"{_display_name(kind)} sent to the model"
    if event_type == "request_completed":
        latency = event.get("latency_s")
        suffix = f" in {float(latency):.2f}s" if isinstance(latency, (int, float)) else ""
        return f"Model response validated{suffix}"
    if event_type == "request_failed":
        return f"{event.get('error_type', 'Request')} failed: {event.get('error', 'unknown error')}"
    if event_type in ("primitive_transition", "skill_transition", "action_transition"):
        return str(event.get("transition", "Controller phase changed"))
    if event_type in ("semantic_skill_outcome", "action_outcome"):
        decision = event.get("action", event.get("skill"))
        return f"{_display_name(decision)}: {event.get('outcome', 'finished')}"
    if event_type == "task_success":
        return "Physical task success contract satisfied"
    if event_type == "request_budget_exhausted":
        return "LLM request budget exhausted; controller stopped safely"
    if event_type == "dashboard_server_started":
        return f"Dashboard published at {event.get('url', 'loopback')}"
    if event_type == "control_sample":
        return f"Control sample at step {event.get('step', '—')}"
    if event_type == "run_started":
        return "Run and durable trace started"
    if event_type == "run_finished":
        return f"Run finished: {event.get('status', 'finished')}"
    return _display_name(event_type)


class _DashboardRequestHandler(SimpleHTTPRequestHandler):
    """Serve the current run quietly, with the dashboard at the root URL."""

    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/":
            self.path = "/dashboard.html"
        super().do_GET()

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        return


class LiveDashboard:
    """Maintain one self-refreshing HTML snapshot and an append-only event log."""

    def __init__(
        self,
        log_root: str | Path,
        *,
        model: str,
        provider: str,
        backend: str,
        refresh_s: float = 0.5,
        input_price_per_million: float | None = None,
        output_price_per_million: float | None = None,
    ) -> None:
        run_name = datetime.now().strftime("%Y%m%d_%H%M%S") + f"_{os.getpid()}"
        self.run_dir = Path(log_root).expanduser().resolve() / run_name
        self.run_dir.mkdir(parents=True, exist_ok=False)
        self.html_path = self.run_dir / "dashboard.html"
        self.jsonl_path = self.run_dir / "events.jsonl"
        self._event_stream = self.jsonl_path.open("a", encoding="utf-8", buffering=1)
        self._refresh_s = float(refresh_s)
        self._input_price = input_price_per_million
        self._output_price = output_price_per_million
        self._http_server: ThreadingHTTPServer | None = None
        self._http_thread: threading.Thread | None = None
        self._recent_events: list[dict[str, Any]] = []
        self._request_traces: list[dict[str, Any]] = []
        self._snapshot: dict[str, Any] = {
            "started_at_utc": _utc_timestamp(),
            "updated_at_utc": _utc_timestamp(),
            "model": model,
            "provider": provider,
            "backend": backend,
            "status": "initializing",
            "control_mode": "not specified",
            "action_space": "not specified",
            "scenario": "not specified",
            "instruction": None,
            "request_count": 0,
            "request_limit": None,
            "latency_s": None,
            "usage": None,
            "estimated_cost_usd": None,
            "visible_reasoning_note": (
                "This page shows the exact prompt, visible model response, parsed plan, and closed-loop decisions. "
                "It cannot display private/hidden chain-of-thought that the model does not return."
            ),
        }
        self.event("run_started", model=model, provider=provider, backend=backend)

    def start_local_server(self, port: int = 8765) -> str:
        """Publish this run on loopback only and return its browser URL."""
        if self._http_server is not None:
            raise RuntimeError("The dashboard HTTP server is already running.")
        handler = partial(_DashboardRequestHandler, directory=str(self.run_dir))
        server = ThreadingHTTPServer(("127.0.0.1", port), handler)
        server.daemon_threads = True
        thread = threading.Thread(
            target=server.serve_forever,
            name="go2-llm-dashboard-http",
            daemon=True,
        )
        self._http_server = server
        self._http_thread = thread
        thread.start()
        actual_port = int(server.server_address[1])
        url = f"http://127.0.0.1:{actual_port}/"
        self.event("dashboard_server_started", bind_host="127.0.0.1", port=actual_port, url=url)
        return url

    def _estimated_cost(self, usage: dict[str, Any] | None) -> float | None:
        if not usage or self._input_price is None or self._output_price is None:
            return None
        prompt_tokens = usage.get("prompt_tokens")
        completion_tokens = usage.get("completion_tokens")
        if not isinstance(prompt_tokens, int) or not isinstance(completion_tokens, int):
            return None
        return (prompt_tokens * float(self._input_price) + completion_tokens * float(self._output_price)) / 1_000_000.0

    def update(self, **fields: Any) -> None:
        """Update the current browser snapshot without adding a JSONL event."""
        usage = fields.get("usage")
        if usage is not None:
            fields["estimated_cost_usd"] = self._estimated_cost(usage)
        self._snapshot.update(fields)
        if self._request_traces:
            request = self._request_traces[-1]
            for name in ("prompt", "request_observation", "raw_response", "program", "usage", "latency_s"):
                if name in fields:
                    request[name] = fields[name]
            if fields.get("raw_response") is not None or fields.get("program") is not None:
                request["status"] = "completed"
            if usage is not None:
                request["estimated_cost_usd"] = self._estimated_cost(usage)
        self._snapshot["updated_at_utc"] = _utc_timestamp()
        self._render()

    def event(self, event_type: str, **payload: Any) -> None:
        """Append one durable event and refresh the recent-event panel."""
        event = {"timestamp_utc": _utc_timestamp(), "type": event_type, **payload}
        self._event_stream.write(json.dumps(event, ensure_ascii=False, separators=(",", ":")) + "\n")
        self._track_request_event(event)
        self._recent_events.append(event)
        del self._recent_events[:-40]
        self._snapshot["updated_at_utc"] = event["timestamp_utc"]
        self._render()

    def _track_request_event(self, event: dict[str, Any]) -> None:
        """Maintain compact per-request traces for the browser's LLM Calls view."""
        event_type = str(event.get("type"))
        if event_type == "request_submitted":
            request_number = event.get("request_index", event.get("request_count", len(self._request_traces) + 1))
            self._request_traces.append(
                {
                    "number": request_number,
                    "kind": event.get("request_kind", "LLM request"),
                    "status": "pending",
                    "submitted_at_utc": event.get("timestamp_utc"),
                    "prompt": event.get("prompt"),
                    "request_observation": event.get("observation"),
                    "model": event.get("model", self._snapshot.get("model")),
                    "provider": event.get("provider", self._snapshot.get("provider")),
                }
            )
            del self._request_traces[:-20]
            return
        if event_type not in ("request_completed", "request_failed") or not self._request_traces:
            return
        request = self._request_traces[-1]
        request.update(
            {
                key: value
                for key, value in event.items()
                if key in ("latency_s", "usage", "raw_response", "program", "error_type", "error", "request_kind")
            }
        )
        request["status"] = "failed" if event_type == "request_failed" else "completed"
        request["completed_at_utc"] = event.get("timestamp_utc")
        if event.get("usage") is not None:
            request["estimated_cost_usd"] = self._estimated_cost(event["usage"])

    def sample(self, *, observation: dict[str, Any], telemetry: dict[str, Any]) -> None:
        """Refresh live state and append a compact sampled control record."""
        self._snapshot["observation"] = observation
        self._snapshot["telemetry"] = telemetry
        self.event("control_sample", **telemetry)

    def close(self, status: str = "run finished") -> None:
        if not self._event_stream.closed:
            self._snapshot["status"] = status
            self.event("run_finished", status=status)
            self._event_stream.close()
        if self._http_server is not None:
            self._http_server.shutdown()
            self._http_server.server_close()
            if self._http_thread is not None:
                self._http_thread.join(timeout=2.0)
            self._http_server = None
            self._http_thread = None

    def _render_plan(self, plan: Any, status: str, history: Any) -> str:
        if not isinstance(plan, dict) or not isinstance(plan.get("steps"), list) or not plan["steps"]:
            return '<div class="empty-state">No validated semantic plan is available yet.</div>'
        steps = plan["steps"]
        match = re.search(r"step\s+(\d+)/(\d+)", status, flags=re.IGNORECASE)
        active_index = int(match.group(1)) - 1 if match else 0
        completed_count = 0
        if isinstance(history, list):
            completed_count = sum(item.get("outcome") == "completed" for item in history if isinstance(item, dict))
            active_index = min(completed_count, len(steps) - 1)
        status_lower = status.lower()
        if any(token in status_lower for token in ("success", "program complete")):
            completed_count = len(steps)
        cards: list[str] = []
        for index, step in enumerate(steps):
            step = step if isinstance(step, dict) else {"op": str(step)}
            if index < completed_count:
                state, marker = "done", "✓"
            elif index == active_index and not any(
                token in status_lower for token in ("waiting", "pending", "initializing", "failed", "exhausted")
            ):
                state, marker = "active", str(index + 1)
            else:
                state, marker = "queued", str(index + 1)
            op = _display_name(step.get("op", "step"))
            target = step.get("target")
            timeout = step.get("timeout_s")
            metadata = []
            if target:
                metadata.append(f"target: {target}")
            if isinstance(timeout, (int, float)) and timeout:
                metadata.append(f"{float(timeout):g}s timeout")
            subtitle = " · ".join(metadata) or "semantic controller step"
            cards.append(
                f'<div class="plan-step {state}"><div class="step-marker">{marker}</div>'
                f'<div><div class="step-name">{html.escape(op)}</div>'
                f'<div class="step-meta">{html.escape(subtitle)}</div></div></div>'
            )
        return '<div class="plan-rail">' + '<div class="plan-arrow">→</div>'.join(cards) + "</div>"

    def _render_progress(self, task_state: Any) -> str:
        if not isinstance(task_state, dict) or not task_state:
            return '<div class="empty-state">Waiting for the first physical-state sample.</div>'
        priority = (
            "pressed",
            "normalized_depth",
            "opened",
            "hinge_angle_deg",
            "bilateral_contact",
            "left_finger_contact",
            "right_finger_contact",
            "lifted",
            "lifted_once",
            "in_tray",
            "any_finger_contact",
            "speed_mps",
        )
        ordered_keys = [key for key in priority if key in task_state]
        ordered_keys.extend(
            key for key in task_state if key not in ordered_keys and not isinstance(task_state[key], dict)
        )
        cards: list[str] = []
        for key in ordered_keys[:10]:
            value = task_state[key]
            if isinstance(value, bool):
                value_text = "Yes" if value else "No"
                tone = "good" if value else "neutral"
            elif isinstance(value, float):
                value_text = f"{value:.4g}"
                tone = "neutral"
            elif isinstance(value, list) and len(value) > 4:
                value_text = f"[{', '.join(map(str, value[:3]))}, …]"
                tone = "neutral"
            else:
                value_text = str(value)
                tone = "neutral"
            cards.append(
                f'<div class="progress-item"><div class="eyebrow">{html.escape(_display_name(key))}</div>'
                f'<div class="progress-value {tone}">{html.escape(value_text)}</div></div>'
            )
        return '<div class="progress-grid">' + "".join(cards) + "</div>"

    def _render_timeline(self) -> str:
        events = [event for event in self._recent_events if event.get("type") != "control_sample"]
        if not events:
            events = self._recent_events
        if not events:
            return '<div class="empty-state">No execution events yet.</div>'
        rows: list[str] = []
        for event in reversed(events[-12:]):
            timestamp = str(event.get("timestamp_utc", ""))
            clock = timestamp.split("T")[-1].replace("+00:00", "Z") if "T" in timestamp else timestamp
            event_type = str(event.get("type", "event"))
            rows.append(
                f'<li class="timeline-row"><span class="timeline-dot {_event_tone(event_type)}"></span>'
                f'<div class="timeline-copy"><div>{html.escape(_event_summary(event))}</div>'
                f'<div class="timeline-meta">{html.escape(clock)} · {html.escape(_display_name(event_type))}</div></div></li>'
            )
        return '<ol class="timeline">' + "".join(rows) + "</ol>"

    def _render_requests(self) -> str:
        if not self._request_traces:
            return (
                '<div class="empty-state large"><div class="empty-icon">↗</div>'
                "No LLM call has been submitted yet. The robot is settling or deterministic mode is active.</div>"
            )
        cards: list[str] = []
        for offset, request in enumerate(reversed(self._request_traces)):
            number = request.get("number", len(self._request_traces) - offset)
            safe_number = re.sub(r"[^a-zA-Z0-9_-]", "-", str(number))
            status = str(request.get("status", "pending"))
            latency = request.get("latency_s")
            latency_text = f"{float(latency):.2f}s" if isinstance(latency, (int, float)) else "—"
            usage = request.get("usage") or {}
            tokens = usage.get("total_tokens", "—") if isinstance(usage, dict) else "—"
            request_cost = request.get("estimated_cost_usd")
            cost_text = f"${float(request_cost):.8f}" if isinstance(request_cost, (int, float)) else "—"
            error = request.get("error")
            error_html = (
                f'<div class="call-error">{html.escape(str(request.get("error_type", "Request")))}: '
                f"{html.escape(str(error))}</div>"
                if error
                else ""
            )
            cards.append(
                f'<details class="call-card" {"open" if offset == 0 else ""} data-persist="call-{safe_number}">'
                f'<summary><div class="call-title"><span class="status-pill {html.escape(status)}">'
                f"{html.escape(status)}</span><strong>Request #{html.escape(str(number))}</strong>"
                f'<span class="muted">{html.escape(_display_name(request.get("kind")))}</span></div>'
                f'<div class="call-stats"><span>{latency_text}</span><span>{tokens} tokens</span>'
                f'<span>{cost_text}</span><span class="chevron">⌄</span></div></summary>{error_html}'
                '<div class="call-grid">'
                '<section class="trace-pane"><div class="pane-heading"><span>Exact prompt</span>'
                f'<button class="copy-button" data-copy-target="prompt-{safe_number}">Copy</button></div>'
                f'<pre id="prompt-{safe_number}">{_escape_pre(request.get("prompt"))}</pre></section>'
                '<section class="trace-pane"><div class="pane-heading"><span>Visible model response</span>'
                f'<button class="copy-button" data-copy-target="response-{safe_number}">Copy</button></div>'
                f'<pre id="response-{safe_number}">{_escape_pre(request.get("raw_response"))}</pre></section></div>'
                '<div class="trace-details-grid">'
                '<details data-persist="request-program"><summary>Validated semantic program</summary>'
                f"<pre>{_escape_pre(request.get('program'))}</pre></details>"
                '<details data-persist="request-state"><summary>Symbolic observation used for this request</summary>'
                f"<pre>{_escape_pre(request.get('request_observation'))}</pre></details></div></details>"
            )
        return "".join(cards)

    def _render_raw_events(self) -> str:
        if not self._recent_events:
            return '<div class="empty-state">No events recorded.</div>'
        rows: list[str] = []
        for index, event in enumerate(reversed(self._recent_events)):
            event_type = str(event.get("type", "event"))
            timestamp = str(event.get("timestamp_utc", ""))
            rows.append(
                f'<details class="event-card" data-event-type="{html.escape(event_type.lower())}" '
                f'data-persist="event-{index}"><summary><span class="timeline-dot {_event_tone(event_type)}"></span>'
                f"<span><strong>{html.escape(_display_name(event_type))}</strong>"
                f'<small>{html.escape(timestamp)}</small></span><span class="event-summary">'
                f"{html.escape(_event_summary(event))}</span></summary><pre>{_escape_pre(event)}</pre></details>"
            )
        return "".join(rows)

    def _render(self) -> None:
        snapshot = self._snapshot
        status = str(snapshot.get("status") or "initializing")
        status_lower = status.lower()
        if any(token in status_lower for token in ("failed", "error", "exhausted")):
            status_tone = "bad"
        elif any(token in status_lower for token in ("success", "complete")):
            status_tone = "good"
        elif any(token in status_lower for token in ("pending", "step", "request", "active")):
            status_tone = "active"
        else:
            status_tone = "neutral"
        usage = snapshot.get("usage") or {}
        estimated_cost = snapshot.get("estimated_cost_usd")
        cost_text = "—"
        if estimated_cost is not None:
            cost_text = f"${float(estimated_cost):.8f}"
        telemetry = snapshot.get("telemetry") or {}
        program = snapshot.get("program")
        initial_plan = snapshot.get("initial_plan")
        history = snapshot.get("execution_history") or []
        plan = initial_plan or program
        policy_status = str(telemetry.get("policy_status", status))
        task_state = telemetry.get("task_state", telemetry.get("switch", {}))
        mode = str(snapshot.get("control_mode") or "not specified")
        action_space = str(snapshot.get("action_space") or "not specified")
        if action_space == "primitive":
            mode_descriptions = {
                "one-shot": "One bounded primitive horizon; no automatic LLM replan.",
                "reactive": "One real control primitive per call; replan after every outcome.",
                "receding-horizon": "Keep the initial primitive horizon as context; replan after every outcome.",
            }
            mode_description = mode_descriptions.get(mode, "Event-driven validated primitive control.")
            control_contract_text = (
                "The language model composes bounded real control primitives; the 50 Hz controller validates and "
                "applies them."
            )
            planner_output_label = "Control primitive"
        else:
            mode_descriptions = {
                "one-shot": "One initial semantic horizon; no automatic LLM replan.",
                "reactive": "One oracle skill per call; replan after every skill outcome.",
                "receding-horizon": "Keep the initial skill horizon as context; replan after every outcome.",
                "deterministic": "No LLM calls; deterministic mechanics-isolation program.",
            }
            mode_description = mode_descriptions.get(mode, "Event-driven validated semantic planning.")
            control_contract_text = (
                "The language model selects validated semantic skills; the 50 Hz controller owns frames, motion, "
                "contact checks, and safety."
            )
            planner_output_label = "LLM skill"
        latency = snapshot.get("latency_s")
        latency_text = f"{float(latency):.2f}s" if isinstance(latency, (int, float)) else "—"
        total_tokens = usage.get("total_tokens", "—") if isinstance(usage, dict) else "—"
        prompt_tokens = usage.get("prompt_tokens", "—") if isinstance(usage, dict) else "—"
        completion_tokens = usage.get("completion_tokens", "—") if isinstance(usage, dict) else "—"
        request_count = snapshot.get("request_count", 0)
        request_limit = snapshot.get("request_limit")
        request_limit_text = "∞" if request_limit in (None, "unlimited", 0) else str(request_limit)
        request_progress = 0.0
        if isinstance(request_count, (int, float)) and isinstance(request_limit, (int, float)) and request_limit > 0:
            request_progress = min(float(request_count) / float(request_limit) * 100.0, 100.0)
        instruction = snapshot.get("instruction") or "Waiting for task instruction metadata."
        scenario = _display_name(snapshot.get("scenario"))
        task_id = snapshot.get("task_id") or "Go2+D1 service run"
        plan_html = self._render_plan(plan, policy_status, history)
        progress_html = self._render_progress(task_state)
        timeline_html = self._render_timeline()
        requests_html = self._render_requests()
        events_html = self._render_raw_events()
        document = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Go2+D1 Mission Console</title>
  <style>
    :root {{ color-scheme:dark; --bg:#080b12; --surface:#0f1420; --surface-2:#151b29; --surface-3:#1b2333;
      --line:#273146; --line-soft:#1d2535; --text:#f3f6fc; --muted:#8d99ad; --faint:#59657a;
      --accent:#7c8cff; --accent-soft:rgba(124,140,255,.14); --cyan:#46c7e8; --good:#52d69b;
      --good-soft:rgba(82,214,155,.12); --warn:#f6c85f; --bad:#ff6b81; --bad-soft:rgba(255,107,129,.12);
      --shadow:0 18px 60px rgba(0,0,0,.28); }}
    * {{ box-sizing:border-box; }} html {{ background:var(--bg); }} body {{ margin:0; color:var(--text);
      background:radial-gradient(circle at 15% -10%,rgba(124,140,255,.13),transparent 28%),var(--bg);
      font:14px/1.5 Inter,ui-sans-serif,system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; }}
    button,input {{ font:inherit; }} button {{ color:inherit; }} .topbar {{ position:sticky; top:0; z-index:20;
      display:flex; align-items:center; justify-content:space-between; min-height:68px; padding:12px 24px;
      background:rgba(8,11,18,.88); border-bottom:1px solid var(--line-soft); backdrop-filter:blur(18px); }}
    .brand {{ display:flex; align-items:center; gap:12px; }} .brand-mark {{ display:grid; place-items:center; width:36px;
      height:36px; border-radius:11px; background:linear-gradient(145deg,var(--accent),var(--cyan)); color:#071019;
      font-weight:900; box-shadow:0 0 30px rgba(124,140,255,.25); }} .brand-title {{ font-weight:720; letter-spacing:-.01em; }}
    .brand-subtitle,.muted {{ color:var(--muted); }} .brand-subtitle {{ font-size:12px; }} .top-actions {{ display:flex;
      align-items:center; gap:10px; }} .live-indicator {{ display:flex; align-items:center; gap:7px; color:var(--muted);
      font-size:12px; }} .live-dot {{ width:8px; height:8px; border-radius:50%; background:var(--good);
      box-shadow:0 0 0 5px rgba(82,214,155,.09); animation:pulse 1.8s infinite; }}
    @keyframes pulse {{ 50% {{ opacity:.45; }} }} .button {{ border:1px solid var(--line); background:var(--surface-2);
      border-radius:9px; padding:7px 11px; cursor:pointer; transition:.15s ease; }} .button:hover {{ border-color:#465570;
      transform:translateY(-1px); }} .app {{ width:min(1680px,100%); margin:0 auto; padding:20px 24px 42px; }}
    .tabbar {{ display:flex; gap:5px; margin-bottom:18px; padding:4px; width:max-content; max-width:100%; overflow:auto;
      background:var(--surface); border:1px solid var(--line-soft); border-radius:11px; }} .tab-button {{ border:0;
      background:transparent; color:var(--muted); padding:8px 13px; border-radius:8px; cursor:pointer; white-space:nowrap; }}
    .tab-button.active {{ color:var(--text); background:var(--surface-3); box-shadow:0 1px 4px rgba(0,0,0,.3); }}
    .tab-panel {{ display:none; }} .tab-panel.active {{ display:block; }} .mission {{ display:grid;
      grid-template-columns:minmax(0,1.6fr) minmax(300px,.7fr); gap:14px; margin-bottom:14px; }} .panel {{ background:linear-gradient(180deg,
      rgba(21,27,41,.96),rgba(15,20,32,.96)); border:1px solid var(--line); border-radius:14px; box-shadow:var(--shadow); }}
    .panel-pad {{ padding:18px; }} .mission-main {{ position:relative; overflow:hidden; min-height:182px; }}
    .mission-main:after {{ content:""; position:absolute; width:240px; height:240px; right:-90px; top:-110px;
      border-radius:50%; background:rgba(124,140,255,.09); filter:blur(5px); }} .eyebrow {{ color:var(--muted);
      font-size:11px; font-weight:700; text-transform:uppercase; letter-spacing:.09em; }} .mission h1 {{ position:relative;
      z-index:1; margin:9px 0 7px; max-width:850px; font-size:clamp(22px,3vw,36px); line-height:1.13;
      letter-spacing:-.035em; }} .mission-copy {{ position:relative; z-index:1; max-width:900px; color:#bec7d7;
      font-size:15px; }} .status-row {{ display:flex; align-items:center; gap:10px; margin-top:20px; }} .status-pill {{ display:inline-flex;
      align-items:center; width:max-content; border:1px solid var(--line); border-radius:99px; padding:3px 8px; color:var(--muted);
      background:var(--surface-3); font-size:11px; font-weight:750; text-transform:uppercase; letter-spacing:.06em; }}
    .status-pill.good,.status-pill.completed {{ color:var(--good); border-color:rgba(82,214,155,.3); background:var(--good-soft); }}
    .status-pill.bad,.status-pill.failed {{ color:var(--bad); border-color:rgba(255,107,129,.3); background:var(--bad-soft); }}
    .status-pill.active,.status-pill.pending {{ color:#aeb7ff; border-color:rgba(124,140,255,.35); background:var(--accent-soft); }}
    .status-text {{ font-weight:700; overflow-wrap:anywhere; }} .mode-card {{ display:flex; flex-direction:column; justify-content:space-between; }}
    .mode-name {{ margin:7px 0 4px; font-size:23px; font-weight:760; letter-spacing:-.025em; }} .mode-flow {{ display:flex;
      align-items:center; gap:7px; flex-wrap:wrap; margin:18px 0 8px; }} .flow-node {{ padding:6px 9px; background:var(--surface-3);
      border:1px solid var(--line); border-radius:8px; font-size:12px; }} .flow-arrow {{ color:var(--faint); }}
    .kpi-grid {{ display:grid; grid-template-columns:repeat(6,minmax(130px,1fr)); gap:10px; margin-bottom:14px; }}
    .kpi {{ padding:14px; background:var(--surface); border:1px solid var(--line-soft); border-radius:12px; }} .kpi-value {{ margin-top:5px;
      font-size:20px; font-weight:750; letter-spacing:-.025em; overflow-wrap:anywhere; }} .kpi-sub {{ margin-top:3px; color:var(--faint); font-size:11px; }}
    .request-bar {{ height:4px; margin-top:10px; background:#242c3c; border-radius:99px; overflow:hidden; }} .request-bar span {{ display:block;
      height:100%; background:linear-gradient(90deg,var(--accent),var(--cyan)); }} .overview-grid {{ display:grid;
      grid-template-columns:minmax(0,1.65fr) minmax(320px,.75fr); gap:14px; }} .section-heading {{ display:flex;
      justify-content:space-between; align-items:flex-start; gap:12px; margin-bottom:14px; }} .section-heading h2 {{ margin:0;
      font-size:16px; letter-spacing:-.015em; }} .section-note {{ color:var(--muted); font-size:12px; }} .plan-rail {{ display:flex;
      align-items:stretch; gap:8px; overflow-x:auto; padding:2px 0 8px; }} .plan-step {{ display:flex; align-items:center; gap:10px;
      min-width:180px; flex:1; padding:11px; border:1px solid var(--line); border-radius:10px; background:var(--surface); }}
    .plan-step.active {{ border-color:rgba(124,140,255,.75); background:var(--accent-soft); box-shadow:0 0 0 1px rgba(124,140,255,.15); }}
    .plan-step.done {{ border-color:rgba(82,214,155,.28); background:var(--good-soft); }} .plan-step.queued {{ opacity:.63; }}
    .step-marker {{ display:grid; place-items:center; flex:0 0 28px; height:28px; border-radius:9px; background:var(--surface-3);
      color:var(--muted); font-size:12px; font-weight:800; }} .active .step-marker {{ color:white; background:var(--accent); }}
    .done .step-marker {{ color:#071a12; background:var(--good); }} .step-name {{ font-weight:730; }} .step-meta {{ color:var(--muted);
      font-size:11px; }} .plan-arrow {{ display:grid; place-items:center; color:var(--faint); }} .active-controller {{ margin-top:12px;
      padding:11px 13px; border-left:3px solid var(--accent); background:rgba(124,140,255,.07); border-radius:0 9px 9px 0; }}
    .controller-value {{ margin-top:2px; font-weight:700; }} .progress-grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(115px,1fr));
      gap:8px; }} .progress-item {{ min-height:72px; padding:11px; border:1px solid var(--line-soft); border-radius:10px;
      background:rgba(8,11,18,.42); }} .progress-value {{ margin-top:7px; font-size:17px; font-weight:760; overflow-wrap:anywhere; }}
    .progress-value.good {{ color:var(--good); }} .timeline {{ list-style:none; margin:0; padding:0; }} .timeline-row {{ display:grid;
      grid-template-columns:12px 1fr; gap:10px; padding:9px 0; border-bottom:1px solid var(--line-soft); }} .timeline-row:last-child {{ border:0; }}
    .timeline-dot {{ width:8px; height:8px; margin-top:6px; border-radius:50%; background:var(--faint); box-shadow:0 0 0 4px rgba(89,101,122,.09); }}
    .timeline-dot.good {{ background:var(--good); }} .timeline-dot.bad {{ background:var(--bad); }} .timeline-dot.active {{ background:var(--accent); }}
    .timeline-copy {{ min-width:0; }} .timeline-meta {{ margin-top:2px; color:var(--faint); font-size:10px; text-transform:uppercase;
      letter-spacing:.055em; }} .notice {{ margin-top:14px; padding:11px 13px; color:#c4ccdb; background:rgba(246,200,95,.06);
      border:1px solid rgba(246,200,95,.18); border-radius:10px; font-size:12px; }} .call-card,.event-card,.trace-details-grid details,
    .raw-card {{ margin-bottom:10px; background:var(--surface); border:1px solid var(--line); border-radius:12px; overflow:hidden; }}
    details > summary {{ list-style:none; cursor:pointer; }} details > summary::-webkit-details-marker {{ display:none; }} .call-card > summary {{ display:flex;
      justify-content:space-between; align-items:center; gap:14px; padding:13px 15px; }} .call-title,.call-stats {{ display:flex;
      align-items:center; gap:9px; }} .call-stats {{ color:var(--muted); font-size:12px; }} .chevron {{ color:var(--faint); font-size:18px; }}
    .call-error {{ margin:0 15px 12px; padding:10px; color:var(--bad); background:var(--bad-soft); border-radius:8px; }}
    .call-grid {{ display:grid; grid-template-columns:1fr 1fr; gap:1px; border-top:1px solid var(--line); background:var(--line); }}
    .trace-pane {{ min-width:0; padding:14px; background:var(--surface-2); }} .pane-heading {{ display:flex; justify-content:space-between;
      align-items:center; margin-bottom:9px; font-weight:700; }} .copy-button {{ border:1px solid var(--line); background:var(--surface-3);
      color:var(--muted); border-radius:7px; padding:4px 8px; cursor:pointer; font-size:11px; }} pre {{ margin:0; padding:13px;
      max-height:460px; overflow:auto; white-space:pre-wrap; overflow-wrap:anywhere; color:#d8e0ef; background:#090d15;
      border:1px solid #1b2433; border-radius:9px; font:12px/1.55 ui-monospace,SFMono-Regular,Consolas,monospace; }}
    .trace-details-grid {{ display:grid; grid-template-columns:1fr 1fr; gap:10px; padding:12px; }} .trace-details-grid details > summary,
    .raw-card > summary {{ padding:11px 13px; font-weight:700; }} .trace-details-grid details pre,.raw-card pre {{ margin:0 12px 12px; }}
    .robot-grid {{ display:grid; grid-template-columns:1fr 1fr; gap:14px; }} .raw-card {{ box-shadow:none; }} .raw-card h2 {{ margin:0;
      padding:13px 15px; border-bottom:1px solid var(--line); font-size:14px; }} .raw-card > pre {{ margin:12px; }} .event-toolbar {{ display:flex;
      align-items:center; justify-content:space-between; gap:12px; margin-bottom:12px; }} .search {{ width:min(420px,100%); padding:9px 11px;
      color:var(--text); background:var(--surface); border:1px solid var(--line); border-radius:9px; outline:none; }} .search:focus {{ border-color:var(--accent); }}
    .event-card > summary {{ display:grid; grid-template-columns:12px minmax(180px,.35fr) 1fr; gap:10px; align-items:start; padding:11px 13px; }}
    .event-card small {{ display:block; margin-top:2px; color:var(--faint); font-weight:400; }} .event-summary {{ color:#b7c0d0; }}
    .empty-state {{ padding:22px; color:var(--muted); text-align:center; border:1px dashed var(--line); border-radius:10px; }}
    .empty-state.large {{ display:grid; place-items:center; min-height:220px; }} .empty-icon {{ font-size:28px; color:var(--accent); }} .footer {{ display:flex;
      justify-content:space-between; gap:20px; margin-top:16px; color:var(--faint); font-size:11px; overflow-wrap:anywhere; }}
    .good {{ color:var(--good); }} .bad {{ color:var(--bad); }}
    @media (max-width:1050px) {{ .mission,.overview-grid {{ grid-template-columns:1fr; }} .kpi-grid {{ grid-template-columns:repeat(3,1fr); }} }}
    @media (max-width:720px) {{ .topbar {{ padding:10px 14px; }} .app {{ padding:14px; }} .live-indicator {{ display:none; }}
      .kpi-grid {{ grid-template-columns:repeat(2,1fr); }} .call-grid,.trace-details-grid,.robot-grid {{ grid-template-columns:1fr; }}
      .call-card > summary,.event-toolbar {{ align-items:flex-start; flex-direction:column; }} .call-stats {{ flex-wrap:wrap; }}
      .event-card > summary {{ grid-template-columns:12px 1fr; }} .event-summary {{ grid-column:2; }} }}
  </style>
</head>
<body>
  <header class="topbar">
    <div class="brand"><div class="brand-mark">G2</div><div><div class="brand-title">Go2+D1 Mission Console</div>
      <div class="brand-subtitle">LLM planning × physical execution trace</div></div></div>
    <div class="top-actions"><div class="live-indicator"><span class="live-dot"></span><span>Live · updated
      {html.escape(str(snapshot.get("updated_at_utc")))}</span></div><button class="button" id="toggle-refresh">Pause refresh</button></div>
  </header>
  <main class="app">
    <nav class="tabbar" aria-label="Dashboard views">
      <button class="tab-button active" data-tab="overview">Overview</button>
      <button class="tab-button" data-tab="calls">LLM calls <span class="status-pill">{len(self._request_traces)}</span></button>
      <button class="tab-button" data-tab="robot">Robot state</button>
      <button class="tab-button" data-tab="events">Raw events</button>
    </nav>

    <section class="tab-panel active" data-panel="overview">
      <div class="mission">
        <article class="panel panel-pad mission-main"><div class="eyebrow">{html.escape(str(task_id))} · {html.escape(scenario)}</div>
          <h1>{html.escape(str(instruction))}</h1><div class="mission-copy">{html.escape(control_contract_text)}</div><div class="status-row">
          <span class="status-pill {status_tone}">{status_tone}</span><span class="status-text">{html.escape(status)}</span></div></article>
        <aside class="panel panel-pad mode-card"><div><div class="eyebrow">LLM mode · {html.escape(_display_name(action_space))} action space</div><div class="mode-name">{html.escape(_display_name(mode))}</div>
          <div class="muted">{html.escape(mode_description)}</div></div><div class="mode-flow"><span class="flow-node">Symbolic state</span>
          <span class="flow-arrow">→</span><span class="flow-node">{html.escape(planner_output_label)}</span><span class="flow-arrow">→</span>
          <span class="flow-node">50 Hz execution</span></div></aside>
      </div>

      <div class="kpi-grid">
        <div class="kpi"><div class="eyebrow">Simulation</div><div class="kpi-value">{html.escape(str(telemetry.get("sim_time_s", "—")))}s</div>
          <div class="kpi-sub">step {html.escape(str(telemetry.get("step", "—")))}</div></div>
        <div class="kpi"><div class="eyebrow">LLM requests</div><div class="kpi-value">{html.escape(str(request_count))} / {html.escape(request_limit_text)}</div>
          <div class="request-bar"><span style="width:{request_progress:.2f}%"></span></div></div>
        <div class="kpi"><div class="eyebrow">Last latency</div><div class="kpi-value">{latency_text}</div>
          <div class="kpi-sub">asynchronous request</div></div>
        <div class="kpi"><div class="eyebrow">Tokens</div><div class="kpi-value">{html.escape(str(total_tokens))}</div>
          <div class="kpi-sub">{html.escape(str(prompt_tokens))} in · {html.escape(str(completion_tokens))} out</div></div>
        <div class="kpi"><div class="eyebrow">Last-call cost</div><div class="kpi-value">{cost_text}</div>
          <div class="kpi-sub">when prices are configured</div></div>
        <div class="kpi"><div class="eyebrow">Model</div><div class="kpi-value" style="font-size:14px">{html.escape(str(snapshot.get("model")))}</div>
          <div class="kpi-sub">{html.escape(str(snapshot.get("provider")))} · {html.escape(str(snapshot.get("backend")))}</div></div>
      </div>

      <div class="overview-grid">
        <div><article class="panel panel-pad"><div class="section-heading"><div><h2>Validated execution plan</h2>
          <div class="section-note">Bounded commands only—no model-generated Python or direct joint commands.</div></div></div>{plan_html}
          <div class="active-controller"><div class="eyebrow">What is happening now</div><div class="controller-value">{html.escape(policy_status)}</div></div></article>
          <article class="panel panel-pad" style="margin-top:14px"><div class="section-heading"><div><h2>Physical task progress</h2>
          <div class="section-note">Measured simulator state, not the LLM's claim.</div></div></div>{progress_html}</article></div>
        <aside class="panel panel-pad"><div class="section-heading"><div><h2>Execution timeline</h2>
          <div class="section-note">Newest event first</div></div></div>{timeline_html}</aside>
      </div>
      <div class="notice">{html.escape(str(snapshot.get("visible_reasoning_note")))}</div>
    </section>

    <section class="tab-panel" data-panel="calls"><div class="section-heading"><div><h2>LLM calls</h2>
      <div class="section-note">Scan each turn, then open its exact prompt, visible response, validated program, and request state.</div></div></div>
      {requests_html}</section>

    <section class="tab-panel" data-panel="robot"><div class="robot-grid">
      <article class="raw-card"><h2>Controller telemetry</h2><pre>{_escape_pre(telemetry)}</pre></article>
      <article class="raw-card"><h2>Current live symbolic observation</h2><pre>{_escape_pre(snapshot.get("observation"))}</pre></article>
      <article class="raw-card"><h2>Latest request observation</h2><pre>{_escape_pre(snapshot.get("request_observation"))}</pre></article>
      <article class="raw-card"><h2>Initial horizon and execution history</h2><pre>{_escape_pre({"initial_plan": initial_plan, "execution_history": history})}</pre></article>
      </div></section>

    <section class="tab-panel" data-panel="events"><div class="event-toolbar"><div><h2 style="margin:0">Raw event stream</h2>
      <div class="section-note">Search the latest structured events or download the full append-only trace.</div></div>
      <input class="search" id="event-search" type="search" placeholder="Filter events, e.g. request, skill, success"></div>
      <div id="event-list">{events_html}</div></section>

    <footer class="footer"><span>Refresh: {self._refresh_s:.2f}s · RGB is never sent to the text LLM</span>
      <span>Durable log: <a href="events.jsonl" style="color:var(--accent)">{html.escape(str(self.jsonl_path))}</a></span></footer>
  </main>
  <script>
    (() => {{
      const prefix = `go2-llm-dashboard:${{location.pathname}}`;
      const pauseKey = `${{prefix}}:paused`;
      const scrollKey = `${{prefix}}:scroll`;
      const tabKey = `${{prefix}}:tab`;
      const button = document.getElementById("toggle-refresh");
      let paused = localStorage.getItem(pauseKey) === "1";
      const updateButton = () => {{ button.textContent = paused ? "Resume refresh" : "Pause refresh"; }};
      updateButton();
      const selectTab = (name) => {{
        document.querySelectorAll(".tab-button").forEach(el => el.classList.toggle("active", el.dataset.tab === name));
        document.querySelectorAll(".tab-panel").forEach(el => el.classList.toggle("active", el.dataset.panel === name));
        localStorage.setItem(tabKey, name);
      }};
      document.querySelectorAll(".tab-button").forEach(el => el.addEventListener("click", () => selectTab(el.dataset.tab)));
      const linkedTab = new URLSearchParams(location.search).get("view");
      selectTab(linkedTab || localStorage.getItem(tabKey) || "overview");
      document.querySelectorAll("details").forEach((detail, index) => {{
        const identity = detail.dataset.persist || index;
        const key = `${{prefix}}:detail:${{identity}}`;
        const saved = localStorage.getItem(key);
        if (saved !== null) detail.open = saved === "1";
        detail.addEventListener("toggle", () => localStorage.setItem(key, detail.open ? "1" : "0"));
      }});
      document.querySelectorAll(".copy-button").forEach(copy => copy.addEventListener("click", async event => {{
        event.preventDefault();
        const target = document.getElementById(copy.dataset.copyTarget);
        if (!target) return;
        await navigator.clipboard.writeText(target.innerText);
        const original = copy.textContent;
        copy.textContent = "Copied";
        window.setTimeout(() => copy.textContent = original, 1200);
      }}));
      const search = document.getElementById("event-search");
      if (search) search.addEventListener("input", () => {{
        const query = search.value.trim().toLowerCase();
        document.querySelectorAll(".event-card").forEach(card => {{
          card.style.display = !query || card.innerText.toLowerCase().includes(query) ? "block" : "none";
        }});
      }});
      const savedScroll = sessionStorage.getItem(scrollKey);
      if (savedScroll !== null) window.scrollTo(0, Number(savedScroll));
      button.addEventListener("click", () => {{
        paused = !paused;
        localStorage.setItem(pauseKey, paused ? "1" : "0");
        updateButton();
        if (!paused) location.reload();
      }});
      if (!paused) window.setTimeout(() => {{
        sessionStorage.setItem(scrollKey, String(window.scrollY));
        location.reload();
      }}, {self._refresh_s * 1000.0:.0f});
    }})();
  </script>
</body></html>"""
        temporary_path = self.html_path.with_suffix(".html.tmp")
        temporary_path.write_text(document, encoding="utf-8")
        os.replace(temporary_path, self.html_path)

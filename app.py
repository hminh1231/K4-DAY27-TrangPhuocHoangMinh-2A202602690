"""Exercise 5 — Streamlit approval UI for the HITL PR review agent.

Run with:
    uv run streamlit run app.py

Goal: wrap the LangGraph built in exercises 1–4 in a web UI that adapts to
the confidence bucket of each PR.

Routing thresholds (common/schemas.py):
    > 72%        auto_approve     UI shows a success card; reviewer does nothing
    58 – 72%     human_approval   UI shows Approve / Reject / Edit buttons
    <  58%       escalate         UI shows a question form for the reviewer,
                                  then the refined review for a final confirm

The graph is compiled with a static breakpoint before `commit`, so after the
escalate → synthesize path the reviewer sees the refined review before it is
posted. On the human_approval path the reviewer has already decided, so the
app steps straight through that breakpoint.
"""

from __future__ import annotations

import asyncio
import sys
import uuid
from typing import Any

import streamlit as st
from dotenv import load_dotenv
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.types import Command

from audit.replay import fetch_threads
from common.db import db_path
from exercises.exercise_4_audit import build_graph


load_dotenv()

# The graph nodes log with Rich ("→", "✓"); don't let a non-UTF-8 stdout crash a review.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

LOAD_ONLY = object()   # run_graph sentinel: read the checkpoint, don't invoke
RISK_ICON = {"low": "🟢", "med": "🟡", "high": "🔴"}


# ─── Session state ─────────────────────────────────────────────────────────
if "thread_id" not in st.session_state:
    st.session_state.thread_id = None
if "pr_url" not in st.session_state:
    st.session_state.pr_url = ""
if "view" not in st.session_state:
    st.session_state.view = None   # what to render for the current thread (see _view)


# ─── Drive the graph ───────────────────────────────────────────────────────
def _view(snap) -> dict:
    """Turn a checkpoint snapshot into what the UI should render next."""
    values = snap.values or {}
    for task in snap.tasks:
        if task.interrupts:
            return {"kind": "interrupt", "payload": task.interrupts[0].value, "values": values}
    if snap.next == ("commit",):
        return {"kind": "confirm_refined", "values": values}
    if not snap.next and values.get("final_action"):
        return {"kind": "final", "values": values}
    if not values:
        return {"kind": "empty", "values": values}
    # A run that crashed mid-way: the checkpointer knows where to pick up.
    return {"kind": "incomplete", "next": list(snap.next), "values": values}


async def run_graph(thread_id: str, graph_input: Any = LOAD_ONLY, update: dict | None = None) -> dict:
    """Invoke the graph once (or just load the thread) and return the next view.

    graph_input: initial state dict, Command(resume=...), None to continue from
    a breakpoint, or LOAD_ONLY to read the saved checkpoint without running.
    update: optional state patch applied before invoking (used to discard a
    refined review).
    """
    async with AsyncSqliteSaver.from_conn_string(db_path()) as cp:
        await cp.setup()
        app = build_graph(cp, interrupt_before=["commit"])
        cfg = {"configurable": {"thread_id": thread_id}}

        if update:
            await app.aupdate_state(cfg, update)
        if graph_input is not LOAD_ONLY:
            await app.ainvoke(graph_input, cfg)
        snap = await app.aget_state(cfg)

        # human_approval → commit: the reviewer already chose, no second confirm.
        if snap.next == ("commit",) and not snap.values.get("escalation_answers"):
            await app.ainvoke(None, cfg)
            snap = await app.aget_state(cfg)
        return _view(snap)


def open_thread(thread_id: str) -> None:
    st.session_state.thread_id = thread_id
    st.query_params["thread"] = thread_id
    view = asyncio.run(run_graph(thread_id))
    st.session_state.view = view
    st.session_state.pr_url = view["values"].get("pr_url", "")


def step(graph_input: Any, update: dict | None = None, spinner: str = "Resuming...") -> None:
    """Advance the current thread and store the resulting view, surfacing errors in the UI."""
    try:
        with st.spinner(spinner):
            st.session_state.view = asyncio.run(
                run_graph(st.session_state.thread_id, graph_input, update)
            )
    except Exception as e:  # GitHub / LLM / bad URL — show it instead of a stack trace
        st.session_state.error = f"{type(e).__name__}: {e}"
    st.rerun()


# ─── Page setup ────────────────────────────────────────────────────────────
st.set_page_config(page_title="HITL PR Review", layout="wide")
st.title("HITL PR Review Agent")

# Reloading the page drops session_state; the thread id in the URL brings it back.
if st.session_state.thread_id is None and "thread" in st.query_params:
    open_thread(st.query_params["thread"])


# ─── Sidebar — recent sessions ─────────────────────────────────────────────
with st.sidebar:
    st.header("Recent sessions")
    threads = asyncio.run(fetch_threads(limit=15))
    if not threads:
        st.caption("No sessions yet — run a review.")
    for t in threads:
        pr = t["pr_url"].split("github.com/")[-1]
        label = f"{RISK_ICON.get(t['worst_risk'], '⚪')} {pr} · {t['thread_id'][:8]}"
        current = t["thread_id"] == st.session_state.thread_id
        if st.button(label, key=f"thread_{t['thread_id']}", use_container_width=True,
                     type="primary" if current else "secondary",
                     help=f"{t['events']} events · last {str(t['last_event'])[:19]}"):
            open_thread(t["thread_id"])
            st.rerun()


# ─── Top form — start a new review ─────────────────────────────────────────
with st.form("start"):
    pr_url = st.text_input(
        "PR URL", value=st.session_state.pr_url,
        placeholder="https://github.com/VinUni-AI20k/PR-Demo/pull/1",
    )
    submitted = st.form_submit_button("Run review")


# ─── Renderers per interrupt kind ──────────────────────────────────────────
def render_comments(comments: list) -> None:
    for c in comments:
        c = c if isinstance(c, dict) else c.model_dump()
        st.markdown(f"- **[{c['severity']}]** `{c['file']}:{c.get('line') or '?'}` — {c['body']}")


def render_approval_card(payload: dict) -> dict | None:
    """58–72% bucket: show the LLM review + 3 buttons. Return resume dict or None."""
    conf = payload["confidence"]
    st.subheader(f"🟡 Approval requested — confidence {conf:.0%}")
    st.caption(payload["confidence_reasoning"])
    st.markdown(payload["summary"])
    render_comments(payload.get("comments", []))

    with st.expander("Diff"):
        st.code(payload.get("diff_preview", ""), language="diff")

    feedback = st.text_input("Feedback (optional)", key=f"approval_feedback_{st.session_state.thread_id}")
    col1, col2, col3 = st.columns(3)
    if col1.button("Approve", type="primary", use_container_width=True):
        return {"choice": "approve", "feedback": feedback}
    if col2.button("Reject", use_container_width=True):
        return {"choice": "reject", "feedback": feedback}
    if col3.button("Edit", use_container_width=True):
        if not feedback.strip():
            st.warning("Describe the edit you want in Feedback first.")
            return None
        return {"choice": "edit", "feedback": feedback}
    return None


def render_escalation_card(payload: dict) -> dict | None:
    """< 58% bucket: show risk factors + question form. Return {question: answer} or None."""
    conf = payload["confidence"]
    st.subheader(f"🔴 Strong escalation — confidence {conf:.0%}")
    st.caption(payload["confidence_reasoning"])
    if payload.get("risk_factors"):
        st.error("**Risks:**\n" + "\n".join(f"- {r}" for r in payload["risk_factors"]))
    st.markdown(payload["summary"])

    with st.form(f"escalation_{st.session_state.thread_id}"):
        st.markdown("**The agent needs your input on:**")
        answers = {
            q: st.text_area(q, key=f"q{i}_{st.session_state.thread_id}", height=80)
            for i, q in enumerate(payload["questions"])
        }
        if st.form_submit_button("Submit answers", type="primary"):
            if any(not a.strip() for a in answers.values()):
                st.warning("Please answer every question.")
                return None
            return {q: a.strip() for q, a in answers.items()}
    return None


def render_confirm_refined(values: dict) -> None:
    """After synthesize: show the refined review and let the reviewer post or discard it."""
    a = values["analysis"]
    st.subheader(f"Refined review — confidence {a.confidence:.0%}")
    st.caption(a.confidence_reasoning)
    st.markdown(a.summary)
    render_comments(a.comments)
    with st.expander("Your answers"):
        for q, ans in (values.get("escalation_answers") or {}).items():
            st.markdown(f"**{q}**\n\n> {ans}")

    col1, col2 = st.columns(2)
    if col1.button("Post refined review", type="primary", use_container_width=True):
        step(None, spinner="Posting comment...")
    if col2.button("Discard", use_container_width=True):
        step(None, update={"escalation_answers": None, "human_choice": "reject",
                           "human_feedback": "Refined review discarded by reviewer"})


def render_final(values: dict) -> None:
    action = values.get("final_action", "?")
    a = values.get("analysis")
    pr = values.get("pr_url", st.session_state.pr_url)

    if action == "auto_committed":
        st.success(f"🟢 Auto-approved — confidence {a.confidence:.0%}. "
                   "The agent posted the review; nothing for you to do.")
    elif action == "committed" and values.get("human_choice") == "edit":
        st.success("✓ Review rewritten from your feedback and posted. "
                   f"Feedback: {values.get('human_feedback')}")
    elif action == "committed":
        st.success("✓ Review comment posted.")
    elif action.endswith("commit_failed"):
        st.error("Posting the comment to GitHub failed — check GITHUB_TOKEN scope. "
                 "See the audit trail for details.")
    else:
        st.warning("Rejected — no comment posted.")

    if a is not None:
        st.markdown(a.summary)
        render_comments(a.comments)
    st.link_button("View PR on GitHub", pr)


# ─── Main flow ─────────────────────────────────────────────────────────────
if submitted and pr_url:
    st.session_state.pr_url = pr_url
    st.session_state.thread_id = str(uuid.uuid4())
    st.query_params["thread"] = st.session_state.thread_id
    st.session_state.view = None
    step({"pr_url": pr_url, "thread_id": st.session_state.thread_id},
         spinner="Fetching PR + asking the LLM...")

if err := st.session_state.pop("error", None):
    st.error(err)

view = st.session_state.view
if st.session_state.thread_id and view is not None:
    st.caption(f"thread_id = `{st.session_state.thread_id}`  ·  replay: "
               f"`uv run python -m audit.replay --thread {st.session_state.thread_id}`")

    if view["kind"] == "interrupt":
        payload = view["payload"]
        if payload["kind"] == "approval_request":
            answer = render_approval_card(payload)
        else:
            answer = render_escalation_card(payload)
        if answer is not None:
            step(Command(resume=answer),
                 spinner="Re-analyzing with your answers..." if payload["kind"] == "escalation"
                 else "Resuming...")

    elif view["kind"] == "confirm_refined":
        render_confirm_refined(view["values"])

    elif view["kind"] == "final":
        render_final(view["values"])

    elif view["kind"] == "incomplete":
        st.warning(f"This session stopped before `{', '.join(view['next'])}` finished.")
        if st.button("Continue from last checkpoint", type="primary"):
            step(None)

    else:
        st.info("No saved state for this thread.")

"""Exercise 4 — Structured SQLite audit trail + durable checkpointer.

Zero setup — SQLite stores everything in a single file (`./hitl_audit.db`).
The audit_events schema is created automatically on first connection.

Goals:

1. Use AsyncSqliteSaver so the graph can resume after a crash.
2. Define and emit an `AuditEntry` (common/schemas.py) for every meaningful step,
   so the full session can be replayed.
3. Verify with `uv run python -m audit.replay --thread <id>`.

Approach:
    - Read `node_fetch_pr` below — it is the one fully-worked example. Pay attention
      to *why* each AuditEntry field has the value it does at that step.
    - For every other node, you decide what to log. Field reference:
      `common/schemas.py:AuditEntry`. Helper: `risk_level_for(confidence)`.
    - Implement the one-line body of `audit()`. Everything else (graph wiring,
      checkpointer setup, interrupt/resume loop) is already done for you.

TODOs to complete (10 total): the audit() body, ONE AuditEntry per node for
analyze/route/commit/auto_approve/synthesize, TWO each for human_approval and
escalate (before + after the interrupt).
"""

from __future__ import annotations

import argparse
import asyncio
import os
import time
import uuid

from dotenv import load_dotenv
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt
from rich.console import Console
from rich.markup import escape
from rich.panel import Panel

from common.db import db_conn, db_path, write_audit_event
from common.github import fetch_pr, post_review_comment
from common.llm import get_llm
from common.schemas import (
    AUTO_APPROVE_THRESHOLD,
    ESCALATE_THRESHOLD,
    AuditEntry,
    PRAnalysis,
    ReviewState,
    risk_level_for,
)


console = Console()
AGENT_ID = "pr-review-agent@v0.1"

# Allow-list the Pydantic models stored in graph state, so restoring a
# checkpoint doesn't rely on LangGraph's permissive (soon blocked) default.
CHECKPOINT_SERDE = JsonPlusSerializer(
    allowed_msgpack_modules=[("common.schemas", "PRAnalysis"), ("common.schemas", "ReviewComment")],
)


async def audit(state, entry: AuditEntry) -> None:
    """Write one structured AuditEntry row to the `audit_events` table.

    `thread_id` and `pr_url` are taken from `state` so callers only build
    the entry itself.
    """
    await write_audit_event(thread_id=state["thread_id"], pr_url=state["pr_url"], entry=entry)


async def audit_before_interrupt(state, entry: AuditEntry) -> None:
    """Like audit(), but skip if this pending row was already written.

    A node that calls interrupt() re-runs from the top on resume, so without
    this guard every pre-interrupt row would appear twice in the replay.
    """
    async with db_conn() as conn:
        async with conn.execute(
            "SELECT 1 FROM audit_events WHERE thread_id = ? AND action = ? AND decision = 'pending'",
            (state["thread_id"], entry.action),
        ) as cur:
            if await cur.fetchone() is not None:
                return
    await audit(state, entry)


# ─── Reference example — read this carefully ───────────────────────────────
async def node_fetch_pr(state):
    console.print("[cyan]→ fetch_pr[/cyan]")
    t0 = time.monotonic()
    with console.status("[dim]Fetching PR from GitHub...[/dim]"):
        pr = fetch_pr(state["pr_url"])
    console.print(f"  [green]✓[/green] {len(pr.files_changed)} files, head {pr.head_sha[:7]}")
    # We've only fetched the diff, not analyzed it. So:
    #   - confidence is unknown → 0.0
    #   - risk_level can't be derived from confidence yet → "med" as neutral default
    #   - decision is "pending" — nothing has been decided
    #   - reviewer_id is None — no human is involved at this stage
    #   - reason is a short human-readable summary of what happened
    await audit(state, AuditEntry(
        agent_id=AGENT_ID,
        action="fetch_pr",
        confidence=0.0,
        risk_level="med",
        decision="pending",
        reason=f"Fetched {len(pr.files_changed)} files, head={pr.head_sha[:7]}",
        execution_time_ms=int((time.monotonic() - t0) * 1000),
    ))
    return {
        "pr_title": pr.title,
        "pr_diff": pr.diff,
        "pr_files": pr.files_changed,
        "pr_head_sha": pr.head_sha,
    }
# ───────────────────────────────────────────────────────────────────────────


async def node_analyze(state):
    console.print("[cyan]→ analyze[/cyan]")
    t0 = time.monotonic()
    llm = get_llm().with_structured_output(PRAnalysis)
    with console.status("[dim]LLM reviewing the diff...[/dim]"):
        a: PRAnalysis = await llm.ainvoke([
            {"role": "system", "content": (
                "You are a senior code reviewer. Review the PR diff and return structured output. "
                "Calibrate confidence honestly: high (>0.8) only for trivial, clearly safe changes; "
                "medium (0.6-0.75) for small features with open questions; "
                "low (<0.55) when there are security, data-integrity or untested risky changes. "
                "If your confidence is below 0.6, populate escalation_questions with 2-4 specific, "
                "context-rich questions for a human reviewer. Each question must reference the file "
                "(and function or line where possible) it is about and state what answer would "
                "resolve your uncertainty. Otherwise leave escalation_questions empty."
            )},
            {"role": "user", "content": f"Title: {state['pr_title']}\nDiff:\n{state['pr_diff']}"},
        ])
    console.print(f"  [green]✓[/green] confidence={a.confidence:.0%}, {len(a.comments)} comment(s)")
    # First step with a real confidence → risk_level is derived from it.
    # Nothing is decided yet (route does that), so decision stays "pending".
    await audit(state, AuditEntry(
        agent_id=AGENT_ID,
        action="analyze",
        confidence=a.confidence,
        risk_level=risk_level_for(a.confidence),
        decision="pending",
        reason=a.confidence_reasoning,
        execution_time_ms=int((time.monotonic() - t0) * 1000),
    ))
    return {"analysis": a}


async def node_route(state):
    console.print("[cyan]→ route[/cyan]")
    t0 = time.monotonic()
    c = state["analysis"].confidence
    if c >= AUTO_APPROVE_THRESHOLD:
        decision = "auto_approve"
    elif c < ESCALATE_THRESHOLD:
        decision = "escalate"
    else:
        decision = "human_approval"
    console.print(f"  [green]✓[/green] decision=[bold]{decision}[/bold] (confidence={c:.0%})")
    route_decision = {"auto_approve": "auto", "human_approval": "pending", "escalate": "escalate"}
    await audit(state, AuditEntry(
        agent_id=AGENT_ID,
        action="route",
        confidence=c,
        risk_level=risk_level_for(c),
        decision=route_decision[decision],
        reason=f"Routed to {decision} (thresholds: auto>={AUTO_APPROVE_THRESHOLD:.0%}, "
               f"escalate<{ESCALATE_THRESHOLD:.0%})",
        execution_time_ms=int((time.monotonic() - t0) * 1000),
    ))
    return {"decision": decision}


async def node_human_approval(state):
    t0 = time.monotonic()
    a = state["analysis"]

    await audit_before_interrupt(state, AuditEntry(
        agent_id=AGENT_ID,
        action="human_approval",
        confidence=a.confidence,
        risk_level=risk_level_for(a.confidence),
        decision="pending",
        reason="Waiting for reviewer: approve / reject / edit",
        execution_time_ms=int((time.monotonic() - t0) * 1000),
    ))

    resp = interrupt({
        "kind": "approval_request",
        "pr_url": state["pr_url"],
        "confidence": a.confidence,
        "confidence_reasoning": a.confidence_reasoning,
        "summary": a.summary,
        "comments": [c.model_dump() for c in a.comments],
        "diff_preview": state["pr_diff"][:2000],
    })

    choice = resp.get("choice") or "reject"
    if choice not in ("approve", "reject", "edit"):
        choice = "reject"
    await audit(state, AuditEntry(
        agent_id=AGENT_ID,
        action="human_approval",
        confidence=a.confidence,
        risk_level=risk_level_for(a.confidence),
        reviewer_id=os.environ.get("GITHUB_USER"),
        decision=choice,
        reason=resp.get("feedback") or None,
        execution_time_ms=int((time.monotonic() - t0) * 1000),
    ))
    resp = {**resp, "choice": choice}
    return {"human_choice": resp.get("choice"), "human_feedback": resp.get("feedback")}


async def node_auto_edit(state):
    """Bonus 4 — reviewer chose `edit`: rewrite the review from their feedback, then commit."""
    console.print("[cyan]→ auto_edit[/cyan]")
    t0 = time.monotonic()
    original: PRAnalysis = state["analysis"]
    llm = get_llm().with_structured_output(PRAnalysis)
    with console.status("[dim]LLM rewriting the review from reviewer feedback...[/dim]"):
        edited: PRAnalysis = await llm.ainvoke([
            {"role": "system", "content": (
                "You are a senior code reviewer revising your own review. A human reviewer asked "
                "for changes. Apply their feedback faithfully: drop, add, reword or re-prioritise "
                "comments as they request, and keep everything they did not object to. "
                "Do not invent issues the diff does not support. Keep the original confidence "
                "unless the feedback changes what you know, and leave escalation_questions empty."
            )},
            {"role": "user", "content": (
                f"Title: {state['pr_title']}\n\n"
                f"Your original review:\n{original.model_dump_json(indent=2)}\n\n"
                f"Reviewer feedback:\n{state.get('human_feedback') or '(none)'}\n\n"
                f"Diff:\n{state['pr_diff']}"
            )},
        ])
    console.print(
        f"  [green]✓[/green] {len(original.comments)} → {len(edited.comments)} comment(s)"
    )
    await audit(state, AuditEntry(
        agent_id=AGENT_ID,
        action="auto_edit",
        confidence=edited.confidence,
        risk_level=risk_level_for(edited.confidence),
        reviewer_id=os.environ.get("GITHUB_USER"),
        decision="edit",
        reason=f"Rewrote review from feedback: {state.get('human_feedback')}",
        execution_time_ms=int((time.monotonic() - t0) * 1000),
    ))
    return {"analysis": edited}


def _render_comment_body(state) -> str:
    a = state["analysis"]
    lines = [f"### Automated review (confidence {a.confidence:.0%})", "", a.summary, ""]
    for c in a.comments:
        lines.append(f"- **[{c.severity}]** `{c.file}:{c.line or '?'}` — {c.body}")
    if state.get("human_feedback"):
        lines.append(f"\n_Reviewer note: {state['human_feedback']}_")
    if state.get("escalation_answers"):
        lines.append("\n_Reviewer answered escalation questions:_")
        for q, ans in state["escalation_answers"].items():
            lines.append(f"> **{q}** {ans}")
    return "\n".join(lines)


def _post(state) -> str:
    try:
        post_review_comment(state["pr_url"], _render_comment_body(state))
        console.print(f"  [green]✓[/green] posted comment to {state['pr_url']}")
        return "committed"
    except Exception as e:
        console.print(f"  [red]✗[/red] post failed: {e}")
        return "commit_failed"


async def node_commit(state):
    console.print("[cyan]→ commit[/cyan]")
    t0 = time.monotonic()
    # Two paths converge here:
    #   1. human_approval → commit (only post if approved)
    #      human_approval → auto_edit → commit (post the rewritten review)
    #   2. escalate → synthesize → commit (always post the refined review)
    if state.get("escalation_answers") or state.get("human_choice") in ("approve", "edit"):
        action = _post(state)
    else:
        console.print(f"  [yellow]·[/yellow] skipping comment (choice={state.get('human_choice')})")
        action = "rejected"
    a = state["analysis"]
    if state.get("escalation_answers"):
        decision = "escalate"
    else:
        decision = state.get("human_choice") or "reject"
    await audit(state, AuditEntry(
        agent_id=AGENT_ID,
        action="commit",
        confidence=a.confidence,
        risk_level=risk_level_for(a.confidence),
        reviewer_id=os.environ.get("GITHUB_USER"),
        decision=decision,
        reason=f"{action}: {len(a.comments)} comment(s)"
               + (f" — reviewer note: {state['human_feedback']}" if state.get("human_feedback") else ""),
        execution_time_ms=int((time.monotonic() - t0) * 1000),
    ))
    return {"final_action": action}


async def node_auto_approve(state):
    console.print("[cyan]→ auto_approve[/cyan]  [dim]high confidence — posting directly[/dim]")
    t0 = time.monotonic()
    a = state["analysis"]
    action = _post(state)
    await audit(state, AuditEntry(
        agent_id=AGENT_ID,
        action="auto_approve",
        confidence=a.confidence,
        risk_level=risk_level_for(a.confidence),
        decision="auto",
        reason=f"{action}: {len(a.comments)} comment(s) — {a.confidence_reasoning}",
        execution_time_ms=int((time.monotonic() - t0) * 1000),
    ))
    return {"final_action": f"auto_{action}"}


async def node_escalate(state):
    t0 = time.monotonic()
    a = state["analysis"]
    questions = a.escalation_questions or ["What is the intent of this PR?"]

    await audit_before_interrupt(state, AuditEntry(
        agent_id=AGENT_ID,
        action="escalate",
        confidence=a.confidence,
        risk_level=risk_level_for(a.confidence),
        decision="pending",
        reason=f"Asked reviewer {len(questions)} question(s): " + " | ".join(questions),
        execution_time_ms=int((time.monotonic() - t0) * 1000),
    ))

    answers = interrupt({
        "kind": "escalation",
        "pr_url": state["pr_url"],
        "confidence": a.confidence,
        "confidence_reasoning": a.confidence_reasoning,
        "summary": a.summary,
        "risk_factors": a.risk_factors,
        "questions": questions,
    })

    await audit(state, AuditEntry(
        agent_id=AGENT_ID,
        action="escalate",
        confidence=a.confidence,
        risk_level=risk_level_for(a.confidence),
        reviewer_id=os.environ.get("GITHUB_USER"),
        decision="escalate",
        reason=" | ".join(f"Q: {q} A: {ans}" for q, ans in answers.items()),
        execution_time_ms=int((time.monotonic() - t0) * 1000),
    ))
    return {"escalation_answers": answers}


async def node_synthesize(state):
    console.print("[cyan]→ synthesize[/cyan]")
    t0 = time.monotonic()
    qa = "\n".join(f"Q: {q}\nA: {a}" for q, a in (state.get("escalation_answers") or {}).items())
    llm = get_llm().with_structured_output(PRAnalysis)
    with console.status("[dim]LLM refining review with reviewer answers...[/dim]"):
        refined: PRAnalysis = await llm.ainvoke([
            {"role": "system", "content": "Refine review with reviewer answers."},
            {"role": "user", "content": f"Diff:\n{state['pr_diff']}\n\nQ&A:\n{qa}"},
        ])
    console.print(f"  [green]✓[/green] refined confidence={refined.confidence:.0%}")
    await audit(state, AuditEntry(
        agent_id=AGENT_ID,
        action="synthesize",
        confidence=refined.confidence,
        risk_level=risk_level_for(refined.confidence),
        decision="pending",
        reason=f"confidence {state['analysis'].confidence:.0%} → {refined.confidence:.0%}: "
               f"{refined.confidence_reasoning}",
        execution_time_ms=int((time.monotonic() - t0) * 1000),
    ))
    # node_commit will run next and post the refined review to the PR.
    return {"analysis": refined}


def build_graph(checkpointer, interrupt_before=None):
    g = StateGraph(ReviewState)
    for name, fn in [
        ("fetch_pr", node_fetch_pr), ("analyze", node_analyze), ("route", node_route),
        ("auto_approve", node_auto_approve), ("human_approval", node_human_approval),
        ("commit", node_commit), ("escalate", node_escalate), ("synthesize", node_synthesize),
        ("auto_edit", node_auto_edit),
    ]:
        g.add_node(name, fn)
    g.add_edge(START, "fetch_pr")
    g.add_edge("fetch_pr", "analyze")
    g.add_edge("analyze", "route")
    g.add_conditional_edges(
        "route", lambda s: s["decision"],
        {"auto_approve": "auto_approve", "human_approval": "human_approval", "escalate": "escalate"},
    )
    g.add_edge("auto_approve", END)
    g.add_conditional_edges(
        "human_approval", lambda s: "auto_edit" if s.get("human_choice") == "edit" else "commit",
        ["auto_edit", "commit"],
    )
    g.add_edge("auto_edit", "commit")
    g.add_edge("commit", END)
    g.add_edge("escalate", "synthesize")
    g.add_edge("synthesize", "commit")
    # AsyncSqliteSaver.from_conn_string() takes no serde argument, so set it here.
    checkpointer.serde = CHECKPOINT_SERDE
    return g.compile(checkpointer=checkpointer, interrupt_before=interrupt_before)


def handle_interrupt(payload):
    kind = payload["kind"]
    if kind == "approval_request":
        console.print(Panel.fit(
            escape(payload["summary"]), title=f"conf={payload['confidence']:.0%}", border_style="green",
        ))
        choice = console.input("approve/reject/edit? ").strip().lower()
        return {"choice": choice, "feedback": console.input("Feedback: ").strip()}
    return {q: console.input(escape(f"Q: {q}\nA: ")).strip() for q in payload["questions"]}


async def run(pr_url: str, thread_id: str | None):
    thread_id = thread_id or str(uuid.uuid4())
    console.rule("[bold]Exercise 4 — SQLite audit trail[/bold]")
    console.print(f"[dim]PR: {pr_url}[/dim]")
    console.print(f"[dim]thread_id = {thread_id}[/dim]\n")

    async with AsyncSqliteSaver.from_conn_string(db_path()) as cp:
        await cp.setup()
        app = build_graph(cp)
        cfg = {"configurable": {"thread_id": thread_id}}

        result = await app.ainvoke({"pr_url": pr_url, "thread_id": thread_id}, cfg)
        while "__interrupt__" in result:
            payload = result["__interrupt__"][0].value
            result = await app.ainvoke(Command(resume=handle_interrupt(payload)), cfg)

        console.rule("Final")
        console.print(f"final_action = {result.get('final_action')}")
        console.print(f"\n[dim]Replay:[/dim] uv run python -m audit.replay --thread {thread_id}")


def main():
    load_dotenv()
    p = argparse.ArgumentParser()
    p.add_argument("--pr", required=True)
    p.add_argument("--thread", help="Resume an existing thread")
    args = p.parse_args()
    asyncio.run(run(args.pr, args.thread))


if __name__ == "__main__":
    main()

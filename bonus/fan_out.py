"""Bonus 3 — Multi-reviewer fan-out with the `Send` API.

On the escalate path the same questions go to several reviewers in
parallel. `route` returns one `Send("ask_reviewer", ...)` per reviewer, so
LangGraph runs the `ask_reviewer` tasks in the same superstep; each one
calls `interrupt()`. `collect` only runs once *every* branch has resumed —
the graph waits for all reviewers. Answers are merged by a reducer, then
`synthesize` re-runs the LLM with every reviewer's input.

    route ──Send──▶ ask_reviewer(alice) ─┐
          └─Send──▶ ask_reviewer(bob)   ─┴─▶ collect ─▶ synthesize ─▶ commit

The CLI resumes one reviewer at a time (`Command(resume={interrupt_id: answers})`)
to show that the graph stays paused until the last reviewer has answered.

Usage:
    uv run python -m bonus.fan_out --pr https://github.com/VinUni-AI20k/PR-Demo/pull/2
    uv run python -m bonus.fan_out --pr ... --reviewers alice,bob,carol

Non-escalate PRs follow the normal exercise 4 flow. Dry-run by default; --post to post.
"""

from __future__ import annotations

import argparse
import asyncio
import time
import uuid
from typing import Annotated

from dotenv import load_dotenv
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, Send, interrupt
from rich.markup import escape
from rich.panel import Panel

import exercises.exercise_4_audit as ex4
from bonus import console, enable_dry_run
from common.db import db_conn, db_path
from common.schemas import AuditEntry, ReviewState, risk_level_for


def merge_answers(old: dict | None, new: dict | None) -> dict:
    """Reducer: parallel ask_reviewer branches each add their own key."""
    return {**(old or {}), **(new or {})}


class FanOutState(ReviewState, total=False):
    reviewers: list[str]
    reviewer: str                                                   # set per Send branch
    reviewer_answers: Annotated[dict[str, dict[str, str]], merge_answers]


def route_after(state: FanOutState):
    if state["decision"] != "escalate":
        return state["decision"]
    return [
        Send("ask_reviewer", {
            "pr_url": state["pr_url"], "thread_id": state["thread_id"],
            "analysis": state["analysis"], "reviewer": r,
        })
        for r in state["reviewers"]
    ]


async def _already_pending(thread_id: str, reviewer: str) -> bool:
    # ask_reviewer re-runs from the top on resume (and when a *sibling* resumes),
    # so write each reviewer's pending row once.
    async with db_conn() as conn:
        async with conn.execute(
            "SELECT 1 FROM audit_events WHERE thread_id = ? AND action = 'escalate' "
            "AND decision = 'pending' AND reviewer_id = ?",
            (thread_id, reviewer),
        ) as cur:
            return await cur.fetchone() is not None


async def node_ask_reviewer(state: FanOutState) -> dict:
    t0 = time.monotonic()
    a, reviewer = state["analysis"], state["reviewer"]
    questions = a.escalation_questions or ["What is the intent of this PR?"]

    if not await _already_pending(state["thread_id"], reviewer):
        await ex4.audit(state, AuditEntry(
            agent_id=ex4.AGENT_ID, action="escalate",
            confidence=a.confidence, risk_level=risk_level_for(a.confidence),
            reviewer_id=reviewer, decision="pending",
            reason=f"Asked {reviewer} {len(questions)} question(s): " + " | ".join(questions),
            execution_time_ms=int((time.monotonic() - t0) * 1000),
        ))

    answers = interrupt({
        "kind": "escalation", "reviewer": reviewer, "pr_url": state["pr_url"],
        "confidence": a.confidence, "confidence_reasoning": a.confidence_reasoning,
        "summary": a.summary, "risk_factors": a.risk_factors, "questions": questions,
    })

    await ex4.audit(state, AuditEntry(
        agent_id=ex4.AGENT_ID, action="escalate",
        confidence=a.confidence, risk_level=risk_level_for(a.confidence),
        reviewer_id=reviewer, decision="escalate",
        reason=" | ".join(f"Q: {q} A: {ans}" for q, ans in answers.items()),
        execution_time_ms=int((time.monotonic() - t0) * 1000),
    ))
    return {"reviewer_answers": {reviewer: answers}}


async def node_collect(state: FanOutState) -> dict:
    """Join point: runs once, after every reviewer branch has answered."""
    console.print("[cyan]→ collect[/cyan]")
    t0 = time.monotonic()
    by_reviewer = state["reviewer_answers"]
    questions = next(iter(by_reviewer.values())).keys()
    # Same shape exercise 4 expects ({question: answer}), one line per reviewer.
    merged = {
        q: "\n".join(f"[{r}] {answers.get(q) or '(no answer)'}" for r, answers in by_reviewer.items())
        for q in questions
    }
    console.print(f"  [green]✓[/green] answers from {', '.join(by_reviewer)}")
    a = state["analysis"]
    await ex4.audit(state, AuditEntry(
        agent_id=ex4.AGENT_ID, action="collect",
        confidence=a.confidence, risk_level=risk_level_for(a.confidence),
        reviewer_id=",".join(by_reviewer), decision="escalate",
        reason=f"Merged answers from {len(by_reviewer)} reviewer(s)",
        execution_time_ms=int((time.monotonic() - t0) * 1000),
    ))
    return {"escalation_answers": merged}


def build_graph(checkpointer):
    g = StateGraph(FanOutState)
    for name, fn in [
        ("fetch_pr", ex4.node_fetch_pr), ("analyze", ex4.node_analyze), ("route", ex4.node_route),
        ("auto_approve", ex4.node_auto_approve), ("human_approval", ex4.node_human_approval),
        ("auto_edit", ex4.node_auto_edit), ("commit", ex4.node_commit),
        ("ask_reviewer", node_ask_reviewer), ("collect", node_collect),
        ("synthesize", ex4.node_synthesize),
    ]:
        g.add_node(name, fn)
    g.add_edge(START, "fetch_pr")
    g.add_edge("fetch_pr", "analyze")
    g.add_edge("analyze", "route")
    g.add_conditional_edges("route", route_after, ["auto_approve", "human_approval", "ask_reviewer"])
    g.add_edge("auto_approve", END)
    g.add_conditional_edges(
        "human_approval", lambda s: "auto_edit" if s.get("human_choice") == "edit" else "commit",
        ["auto_edit", "commit"],
    )
    g.add_edge("auto_edit", "commit")
    g.add_edge("ask_reviewer", "collect")
    g.add_edge("collect", "synthesize")
    g.add_edge("synthesize", "commit")
    g.add_edge("commit", END)
    return g.compile(checkpointer=checkpointer)


def ask(payload: dict) -> dict:
    reviewer = payload["reviewer"]
    console.print(Panel.fit(
        escape(payload["summary"]) + "\n\n[bold]Risks:[/bold]\n"
        + "\n".join(f"- {escape(r)}" for r in payload["risk_factors"]),
        title=f"Escalation for [bold]{reviewer}[/bold] — conf={payload['confidence']:.0%}",
        border_style="yellow",
    ))
    return {q: console.input(escape(f"[{reviewer}] Q: {q}\nA: ")).strip() for q in payload["questions"]}


async def run(pr_url: str, reviewers: list[str], post: bool) -> None:
    if not post:
        enable_dry_run()
    thread_id = str(uuid.uuid4())
    cfg = {"configurable": {"thread_id": thread_id}}
    console.rule("[bold]Bonus 3 — multi-reviewer fan-out[/bold]")
    console.print(f"[dim]PR: {pr_url}\nreviewers: {', '.join(reviewers)}\nthread_id = {thread_id}[/dim]\n")

    async with AsyncSqliteSaver.from_conn_string(db_path()) as cp:
        await cp.setup()
        app = build_graph(cp)
        result = await app.ainvoke({"pr_url": pr_url, "thread_id": thread_id, "reviewers": reviewers}, cfg)

        while "__interrupt__" in result:
            pending = result["__interrupt__"]
            if pending[0].value["kind"] == "approval_request":
                result = await app.ainvoke(Command(resume=ex4.handle_interrupt(pending[0].value)), cfg)
                continue
            waiting = [i.value["reviewer"] for i in pending]
            console.print(f"\n[yellow]⏸ waiting for {len(waiting)} reviewer(s): {', '.join(waiting)}[/yellow]")
            first = pending[0]
            result = await app.ainvoke(Command(resume={first.id: ask(first.value)}), cfg)

        console.rule("Final")
        console.print(f"final_action = {result.get('final_action')}")
        if result.get("reviewer_answers"):
            console.print(f"answered by  = {', '.join(result['reviewer_answers'])}")
        if result.get("analysis"):
            console.print(f"confidence   = {result['analysis'].confidence:.0%}")
        console.print(f"\n[dim]Replay:[/dim] uv run python -m audit.replay --thread {thread_id}")


def main() -> None:
    load_dotenv()
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pr", required=True)
    p.add_argument("--reviewers", default="alice,bob", help="Comma-separated reviewer ids")
    p.add_argument("--post", action="store_true", help="Really post comments to GitHub")
    args = p.parse_args()
    reviewers = [r.strip() for r in args.reviewers.split(",") if r.strip()]
    if len(set(reviewers)) != len(reviewers) or not reviewers:
        p.error("--reviewers must be a non-empty list of distinct ids")
    asyncio.run(run(args.pr, reviewers, args.post))


if __name__ == "__main__":
    main()

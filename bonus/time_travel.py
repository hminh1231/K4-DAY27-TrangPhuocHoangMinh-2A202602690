"""Bonus 1 — Time-travel: fork a session at the HITL checkpoint and compare.

1. Run a review to completion with reviewer answer A (or load an existing
   thread with --thread).
2. List every checkpoint with `app.aget_state_history(config)`.
3. Go back to the checkpoint where the graph was paused at the HITL node,
   fork it, and resume the fork with a different answer B.
4. Show both outcomes side by side.

Why fork instead of `ainvoke(Command(resume=B), old_checkpoint_config)`?
The old checkpoint still holds the pending write of answer A, so resuming
from it just replays A. `aupdate_state(..., as_node="route")` creates a new
child checkpoint with no pending writes — the graph re-enters the HITL node
fresh and `interrupt()` asks again. The LLM analysis is *not* re-run: both
branches start from exactly the same review, only the human input differs.

Usage:
    uv run python -m bonus.time_travel --pr https://github.com/VinUni-AI20k/PR-Demo/pull/1
    uv run python -m bonus.time_travel --pr ... --answer-a approve --answer-b "edit:Drop the nits"
    uv run python -m bonus.time_travel --thread <existing thread_id> --answer-b reject

Answers are `approve`, `reject` or `edit`, optionally followed by `:<feedback>`.
On the escalate path (PR #2) the questions are asked interactively for each branch.
Dry-run by default (comments are printed, not posted); pass --post to post.
"""

from __future__ import annotations

import argparse
import asyncio
import uuid

from dotenv import load_dotenv
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.types import Command
from rich.columns import Columns
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

import exercises.exercise_4_audit as ex4
from bonus import console, enable_dry_run
from common.db import db_path


HITL_NODES = {("human_approval",), ("escalate",)}


def parse_answer(raw: str) -> dict:
    choice, _, feedback = raw.partition(":")
    choice = choice.strip().lower()
    if choice not in ("approve", "reject", "edit"):
        raise argparse.ArgumentTypeError(f"answer must be approve/reject/edit[:feedback], got {raw!r}")
    return {"choice": choice, "feedback": feedback.strip()}


def answer_for(payload: dict, scripted: dict, label: str) -> dict:
    """Scripted answer on the approval path; ask the human on the escalate path."""
    if payload["kind"] == "approval_request":
        console.print(f"  [magenta]branch {label}[/magenta] reviewer answers: {scripted}")
        return scripted
    console.print(f"\n[bold yellow]Branch {label}[/bold yellow] — answer the escalation questions:")
    return ex4.handle_interrupt(payload)


async def drive(app, graph_input, cfg, scripted: dict, label: str) -> dict:
    result = await app.ainvoke(graph_input, cfg)
    while "__interrupt__" in result:
        payload = result["__interrupt__"][0].value
        result = await app.ainvoke(Command(resume=answer_for(payload, scripted, label)), cfg)
    return result


def print_history(history) -> None:
    table = Table(title="Checkpoint history (newest first)")
    for col in ("step", "checkpoint_id", "source", "next", "decision", "human_choice", "final_action"):
        table.add_column(col)
    for snap in history:
        v = snap.values or {}
        paused = any(t.interrupts for t in snap.tasks)
        table.add_row(
            str(snap.metadata.get("step")),
            snap.config["configurable"]["checkpoint_id"][-12:],
            str(snap.metadata.get("source")),
            ", ".join(snap.next) + ("  ⏸" if paused else ""),
            str(v.get("decision") or "-"),
            str(v.get("human_choice") or "-"),
            str(v.get("final_action") or "-"),
        )
    console.print(table)


def outcome_panel(label: str, values: dict) -> Panel:
    a = values.get("analysis")
    lines = [
        f"human_choice   = {values.get('human_choice') or '-'}",
        f"human_feedback = {values.get('human_feedback') or '-'}",
        f"answers        = {len(values.get('escalation_answers') or {})}",
        f"confidence     = {a.confidence:.0%}" if a else "confidence     = -",
        f"comments       = {len(a.comments)}" if a else "comments       = -",
        f"final_action   = {values.get('final_action') or '-'}",
        "",
        ex4._render_comment_body(values) if a else "",
    ]
    return Panel(Text("\n".join(lines)), title=f"Branch {label}", border_style="cyan", width=70)


async def run(args) -> None:
    if not args.post:
        enable_dry_run()

    async with AsyncSqliteSaver.from_conn_string(db_path()) as cp:
        await cp.setup()
        app = ex4.build_graph(cp)

        thread_id = args.thread or str(uuid.uuid4())
        cfg = {"configurable": {"thread_id": thread_id}}
        console.rule("[bold]Bonus 1 — time-travel[/bold]")
        console.print(f"[dim]thread_id = {thread_id}[/dim]\n")

        if args.thread:
            if not (await app.aget_state(cfg)).values:
                console.print(f"[red]No checkpoints for thread {thread_id}[/red]")
                return
        else:
            console.rule("Branch A — original run")
            await drive(app, {"pr_url": args.pr, "thread_id": thread_id}, cfg, args.answer_a, "A")

        history = [snap async for snap in app.aget_state_history(cfg)]
        print_history(history)
        values_a = history[0].values

        # The checkpoint paused at the HITL node; the oldest one is the original branch.
        paused = [s for s in history if s.next in HITL_NODES]
        if not paused:
            console.print("[yellow]This session never paused for a human (auto_approve path) — "
                          "nothing to fork. Try PR #1 or PR #2.[/yellow]")
            return
        fork_point = paused[-1]
        console.print(
            f"\nForking from step {fork_point.metadata.get('step')} "
            f"(checkpoint …{fork_point.config['configurable']['checkpoint_id'][-12:]}, "
            f"next={fork_point.next[0]})"
        )

        console.rule("Branch B — forked run")
        fork_cfg = await app.aupdate_state(
            fork_point.config, {"decision": fork_point.values["decision"]}, as_node="route",
        )
        await drive(app, None, fork_cfg, args.answer_b, "B")
        values_b = (await app.aget_state(cfg)).values

        console.rule("Comparison")
        console.print(Columns([outcome_panel("A", values_a), outcome_panel("B", values_b)]))
        a_conf, b_conf = values_a["analysis"].confidence, values_b["analysis"].confidence
        console.print(
            f"\nSame starting review, different human input → "
            f"final_action [bold]{values_a.get('final_action')}[/bold] vs "
            f"[bold]{values_b.get('final_action')}[/bold], "
            f"confidence {a_conf:.0%} vs {b_conf:.0%}."
        )
        console.print(f"[dim]Both branches are in the audit trail:[/dim] "
                      f"uv run python -m audit.replay --thread {thread_id}")


def main() -> None:
    load_dotenv()
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--pr", help="Run a fresh review of this PR as branch A")
    src.add_argument("--thread", help="Fork an existing (finished) session instead")
    p.add_argument("--answer-a", type=parse_answer, default=parse_answer("approve"))
    p.add_argument("--answer-b", type=parse_answer,
                   default=parse_answer("edit:Drop the nits; keep only issues and blockers."))
    p.add_argument("--post", action="store_true", help="Really post comments to GitHub")
    asyncio.run(run(p.parse_args()))


if __name__ == "__main__":
    main()

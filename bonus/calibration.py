"""Bonus 2 — Confidence calibration from the audit trail.

Every reviewer decision on the human_approval path is a label for the
model's self-reported confidence: if the model says "70% sure" then, over
many sessions, reviewers should approve its review as-is about 70% of the
time.

    avg confidence  >  approve rate   →  over-confident  (trusts itself too much)
    avg confidence  <  approve rate   →  under-confident (asks humans needlessly)

Reported:
  * the headline query  AVG(confidence) vs COUNT(decision = 'approve')
  * a reliability table per confidence bin
  * Brier score and expected calibration error (ECE)
  * routing mix and how much escalation answers moved confidence

Usage:
    uv run python -m bonus.calibration
    uv run python -m bonus.calibration --edit-credit 0.5   # count `edit` as half-right
"""

from __future__ import annotations

import argparse
import asyncio

from rich.table import Table

from bonus import console
from common.db import db_conn


# One row per human decision on the approval path (forked branches count too —
# each one is a separate reviewer answer to the same confidence).
HEADLINE_SQL = """
SELECT COUNT(*)                                          AS n,
       AVG(confidence)                                   AS avg_confidence,
       SUM(CASE WHEN decision = 'approve' THEN 1 ELSE 0 END) AS approved,
       SUM(CASE WHEN decision = 'edit'    THEN 1 ELSE 0 END) AS edited,
       SUM(CASE WHEN decision = 'reject'  THEN 1 ELSE 0 END) AS rejected
  FROM audit_events
 WHERE action = 'human_approval' AND decision IN ('approve', 'edit', 'reject')
"""

SAMPLES_SQL = """
SELECT confidence, decision
  FROM audit_events
 WHERE action = 'human_approval' AND decision IN ('approve', 'edit', 'reject')
"""

ROUTING_SQL = """
SELECT CASE decision WHEN 'auto' THEN 'auto_approve'
                     WHEN 'pending' THEN 'human_approval'
                     ELSE decision END                  AS decision,
       COUNT(*)                                         AS n,
       AVG(confidence)                                  AS avg_confidence
  FROM audit_events
 WHERE action = 'route'
 GROUP BY 1
"""

# Initial confidence at escalate vs refined confidence after synthesize, per thread.
ESCALATION_SQL = """
SELECT e.thread_id, e.confidence AS before, s.confidence AS after
  FROM (SELECT thread_id, MIN(confidence) AS confidence FROM audit_events
         WHERE action = 'escalate' GROUP BY thread_id) e
  JOIN (SELECT thread_id, MAX(id) AS id FROM audit_events
         WHERE action = 'synthesize' GROUP BY thread_id) last ON last.thread_id = e.thread_id
  JOIN audit_events s ON s.id = last.id
"""


async def query(sql: str) -> list[dict]:
    async with db_conn() as conn:
        async with conn.execute(sql) as cur:
            return [dict(r) for r in await cur.fetchall()]


def verdict(gap: float, tolerance: float) -> str:
    if gap > tolerance:
        return f"[red]OVER-confident[/red] by {gap:+.0%} — it claims more certainty than reviewers grant"
    if gap < -tolerance:
        return f"[yellow]UNDER-confident[/yellow] by {gap:+.0%} — reviewers approve more often than it expects"
    return f"[green]well calibrated[/green] (gap {gap:+.0%}, within ±{tolerance:.0%})"


def reliability(samples: list[tuple[float, float]], bin_width: float) -> tuple[Table, float]:
    nbins = round(1 / bin_width)
    bins: dict[int, list[tuple[float, float]]] = {}
    for conf, label in samples:
        bins.setdefault(min(int(conf / bin_width), nbins - 1), []).append((conf, label))

    table = Table(title="Reliability by confidence bin")
    for col in ("bin", "n", "avg confidence", "approve rate", "gap"):
        table.add_column(col, justify="right")
    ece = 0.0
    for b in sorted(bins):
        rows = bins[b]
        avg_c = sum(c for c, _ in rows) / len(rows)
        rate = sum(y for _, y in rows) / len(rows)
        ece += len(rows) / len(samples) * abs(avg_c - rate)
        gap = avg_c - rate
        color = "red" if gap > 0.1 else "yellow" if gap < -0.1 else "green"
        table.add_row(
            f"{b * bin_width:.2f}–{(b + 1) * bin_width:.2f}", str(len(rows)),
            f"{avg_c:.0%}", f"{rate:.0%}", f"[{color}]{gap:+.0%}[/{color}]",
        )
    return table, ece


async def run(args) -> None:
    console.rule("[bold]Bonus 2 — confidence calibration[/bold]")

    head = (await query(HEADLINE_SQL))[0]
    n = head["n"] or 0
    if n == 0:
        console.print("[yellow]No reviewer decisions in audit_events yet. Run a few sessions "
                      "through the human_approval path (PR #1, app.py, or bonus.time_travel) first.[/yellow]")
        return

    credit = {"approve": 1.0, "edit": args.edit_credit, "reject": 0.0}
    samples = [(r["confidence"], credit[r["decision"]]) for r in await query(SAMPLES_SQL)]
    approve_rate = sum(y for _, y in samples) / n
    avg_conf = head["avg_confidence"]

    console.print(f"[dim]{HEADLINE_SQL.strip()}[/dim]\n")
    console.print(
        f"reviewer decisions = [bold]{n}[/bold]  "
        f"(approve {head['approved']}, edit {head['edited']}, reject {head['rejected']})"
    )
    console.print(f"AVG(confidence)    = [bold]{avg_conf:.1%}[/bold]")
    console.print(f"approve rate       = [bold]{approve_rate:.1%}[/bold]"
                  + (f"  (edit counted as {args.edit_credit:g})" if args.edit_credit else ""))
    console.print(f"verdict            = {verdict(avg_conf - approve_rate, args.tolerance)}")
    if n < args.min_sessions:
        console.print(f"[dim]Only {n} decision(s) — fewer than {args.min_sessions}, "
                      "treat the verdict as anecdotal.[/dim]")

    table, ece = reliability(samples, args.bin_width)
    brier = sum((c - y) ** 2 for c, y in samples) / n
    console.print()
    console.print(table)
    console.print(f"Brier score = {brier:.3f}  (0 = perfect, 0.25 = always saying 50%)")
    console.print(f"ECE         = {ece:.1%}  (weighted |confidence − approve rate| over bins)\n")

    routing = await query(ROUTING_SQL)
    if routing:
        t = Table(title="Routing mix")
        for col in ("route decision", "n", "avg confidence"):
            t.add_column(col, justify="right")
        for r in routing:
            t.add_row(r["decision"], str(r["n"]), f"{r['avg_confidence']:.0%}")
        console.print(t)

    esc = await query(ESCALATION_SQL)
    if esc:
        lift = sum(r["after"] - r["before"] for r in esc) / len(esc)
        console.print(f"Escalations: {len(esc)} — reviewer answers moved confidence by "
                      f"[bold]{lift:+.0%}[/bold] on average (escalate → synthesize).")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--edit-credit", type=float, default=0.0,
                   help="How much an `edit` counts as a correct review (0 = wrong, 1 = right)")
    p.add_argument("--bin-width", type=float, default=0.05)
    p.add_argument("--tolerance", type=float, default=0.05,
                   help="|avg confidence − approve rate| still called calibrated")
    p.add_argument("--min-sessions", type=int, default=20)
    asyncio.run(run(p.parse_args()))


if __name__ == "__main__":
    main()

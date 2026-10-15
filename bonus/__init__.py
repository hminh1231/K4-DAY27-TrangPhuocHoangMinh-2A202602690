"""Bonus challenges built on top of the exercise 4 graph.

Every bonus script runs in dry-run mode by default: review comments are
printed instead of posted, so experimenting (forking a session, fanning out
to several reviewers) doesn't spam the demo PRs. Pass `--post` to really post.
"""

from __future__ import annotations

from rich.console import Console
from rich.markup import escape
from rich.panel import Panel
from rich.text import Text

import exercises.exercise_4_audit as ex4


console = Console()


def enable_dry_run() -> list[str]:
    """Replace the GitHub poster used by exercise 4 with a printer.

    Returns the list that collects every comment body "posted" in this
    process, so callers can compare what each branch would have written.
    """
    posted: list[str] = []

    def fake_post(pr_url: str, body: str) -> None:
        posted.append(body)
        console.print(Panel(Text(body), title=escape(f"[dry-run] would post to {pr_url}"), border_style="dim"))

    ex4.post_review_comment = fake_post
    return posted

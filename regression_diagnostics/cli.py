"""Click CLI for regression-diagnostics."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import click

if TYPE_CHECKING:
    from regression_diagnostics.models import DiagnosticsResult


@click.group()
@click.version_option(package_name="regression-diagnostics")
def cli() -> None:
    """OLS regression diagnostics — assumptions, VIF, and influential points."""


@cli.command()
@click.argument("input_file", type=click.Path(exists=True, dir_okay=False))
@click.option("--outcome", "-y", required=True, help="Outcome (dependent) variable column name.")
@click.option("--predictors", "-x", required=True, help="Comma-separated predictor column names.")
@click.option("--cooks-threshold", default=None, type=float, help="Cook's distance threshold (default: 4/n).")
@click.option("--llm-backend", type=click.Choice(["openai", "anthropic", "gemini"]), default=None)
@click.option("--output", "-o", default=None, help="Write JSON report to this path.")
@click.option("--quiet", "-q", is_flag=True)
def check(
    input_file: str,
    outcome: str,
    predictors: str,
    cooks_threshold: Optional[float],
    llm_backend: Optional[str],
    output: Optional[str],
    quiet: bool,
) -> None:
    """Run OLS diagnostics on a CSV file."""
    import pandas as pd

    from regression_diagnostics.engine import run_diagnostics

    df = pd.read_csv(input_file)
    pred_list = [p.strip() for p in predictors.split(",")]

    result = run_diagnostics(df, outcome=outcome, predictors=pred_list,
                             cooks_threshold=cooks_threshold, llm_backend=llm_backend)

    if not quiet:
        _print_result(result)

    if output:
        Path(output).write_text(json.dumps(result.to_dict(), indent=2))
        if not quiet:
            click.echo(f"✓ Report written to {output}")


def _print_result(result: "DiagnosticsResult") -> None:
    try:
        from rich import box
        from rich.console import Console
        from rich.table import Table
        console = Console()
        console.print(f"\n[bold]regression-diagnostics[/] · n={result.n_obs}, "
                      f"p={result.n_predictors}, R²={result.r_squared:.4f}, "
                      f"Adj-R²={result.adj_r_squared:.4f}")
        t = Table(box=box.SIMPLE_HEAVY, header_style="bold")
        t.add_column("Assumption")
        t.add_column("Result")
        t.add_column("Detail")
        colours = {"ok": "green", "warning": "yellow", "violation": "red"}
        for a in result.assumptions:
            c = colours.get(a.severity, "white")
            t.add_row(a.name, f"[{c}]{a.severity.upper()}[/]", a.detail[:70])
        console.print(t)
        if result.vif_results:
            console.print("\n[bold]VIF[/]")
            for v in result.vif_results:
                c = colours.get(v.severity, "white")
                console.print(f"  {v.predictor}: [{c}]{v.vif:.2f}[/]")
        if result.influential_points:
            console.print(f"\n[yellow]Influential points: {len(result.influential_points)}[/]")
        if result.narrative:
            from rich.panel import Panel
            console.print(Panel(result.narrative, title="Narrative", expand=False))
        console.print()
    except ImportError:
        click.echo(f"\nregression-diagnostics: n={result.n_obs}, R²={result.r_squared:.4f}")
        for a in result.assumptions:
            click.echo(f"  {a.name}: {a.severity} — {a.detail[:60]}")

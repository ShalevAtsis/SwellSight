"""
Command-line interface for SwellSight.

Delegates to the maintained scripts under ``scripts/``.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import click

from swellsight.utils.logging import setup_logging

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = REPO_ROOT / "scripts"


def _run_script(script_name: str, extra_args: list[str]) -> int:
    script = SCRIPTS_DIR / script_name
    if not script.exists():
        click.echo(f"Script not found: {script}", err=True)
        return 1
    cmd = [sys.executable, str(script), *extra_args]
    result = subprocess.run(cmd, cwd=str(REPO_ROOT))
    return int(result.returncode)


@click.group()
@click.option("--debug", is_flag=True, help="Enable debug logging")
@click.pass_context
def cli(ctx, debug: bool):
    """SwellSight wave analysis — train, evaluate, and run inference."""
    setup_logging(log_level="DEBUG" if debug else "INFO")
    ctx.ensure_object(dict)
    ctx.obj["debug"] = debug


@cli.command("train")
@click.option("--config", default="configs/training.yaml", show_default=True)
@click.option("--data-dir", default="data", show_default=True)
@click.option("--output-dir", default="outputs/training", show_default=True)
@click.option("--resume", default=None, help="Checkpoint path to resume from")
@click.option("--gpu", type=int, default=None)
@click.pass_context
def train_cmd(ctx, config, data_dir, output_dir, resume, gpu):
    """Train the wave analysis model."""
    args = ["--config", config, "--data-dir", data_dir, "--output-dir", output_dir]
    if resume:
        args.extend(["--resume", resume])
    if gpu is not None:
        args.extend(["--gpu", str(gpu)])
    if ctx.obj.get("debug"):
        args.append("--debug")
    sys.exit(_run_script("train.py", args))


@cli.command("evaluate")
@click.option("--config", default="configs/evaluation.yaml", show_default=True)
@click.option("--model-path", required=True, help="Checkpoint (.pth)")
@click.option("--test-data", required=True, help="Dataset directory")
@click.option("--output-dir", default="outputs/evaluation", show_default=True)
@click.pass_context
def evaluate_cmd(ctx, config, model_path, test_data, output_dir):
    """Evaluate a trained checkpoint on labeled data."""
    args = [
        "--config", config,
        "--model-path", model_path,
        "--test-data", test_data,
        "--output-dir", output_dir,
    ]
    if ctx.obj.get("debug"):
        args.append("--debug")
    sys.exit(_run_script("evaluate.py", args))


@cli.command("analyze")
@click.option("--config", default="configs/inference.yaml", show_default=True)
@click.option("--input", required=True, help="Image file or directory")
@click.option("--output", default="outputs/inference", show_default=True)
@click.option("--checkpoint", default=None, help="Wave model checkpoint (.pth)")
@click.pass_context
def analyze_cmd(ctx, config, input, output, checkpoint):
    """Analyze beach cam images (depth + wave metrics)."""
    args = ["--config", config, "--input", input, "--output", output]
    if checkpoint:
        args.extend(["--checkpoint", checkpoint])
    if ctx.obj.get("debug"):
        args.append("--debug")
    sys.exit(_run_script("inference.py", args))


@cli.command("serve")
@click.option("--host", default="0.0.0.0", show_default=True)
@click.option("--port", default=8000, show_default=True)
def serve_cmd(host, port):
    """Start the REST API server."""
    sys.exit(_run_script("start_api.py", ["--host", host, "--port", str(port)]))


@cli.command("check")
def check_cmd():
    """Verify system readiness for training."""
    sys.exit(_run_script("check_training_readiness.py", []))


def main():
    cli(obj={})


if __name__ == "__main__":
    main()

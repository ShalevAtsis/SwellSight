"""
Command-line interface for SwellSight Wave Analysis System.

Provides a unified CLI for training, evaluation, and inference operations.
"""

import click
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from swellsight.utils.logging import setup_logging

@click.group()
@click.option('--debug', is_flag=True, help='Enable debug logging')
@click.option('--config', type=str, help='Configuration file path')
@click.pass_context
def cli(ctx, debug, config):
    """SwellSight Wave Analysis System CLI."""
    # Setup logging
    log_level = "DEBUG" if debug else "INFO"
    setup_logging(log_level=log_level)
    
    # Store context
    ctx.ensure_object(dict)
    ctx.obj['debug'] = debug
    ctx.obj['config'] = config

@cli.command()
@click.option('--data-dir', type=str, default='data', help='Training data directory')
@click.option('--output-dir', type=str, default='outputs/training', help='Output directory')
@click.option('--resume', type=str, help='Resume from checkpoint')
@click.option('--gpu', type=int, help='GPU device ID')
@click.pass_context
def train(ctx, data_dir, output_dir, resume, gpu):
    """Train wave analysis model."""
    click.echo("🌊 Starting SwellSight training...")
    
    config_path = ctx.obj.get('config', 'configs/training.yaml')
    click.echo(f"Config: {config_path}")
    click.echo(f"Data: {data_dir}")
    click.echo(f"Output: {output_dir}")
    
    # TODO: Implement training in task 7.1
    click.echo("Training pipeline will be implemented in task 7.1")

@cli.command()
@click.option('--model-path', type=str, required=True, help='Trained model path')
@click.option('--test-data', type=str, required=True, help='Test dataset path')
@click.option('--output-dir', type=str, default='outputs/evaluation', help='Output directory')
@click.option('--benchmark', is_flag=True, help='Include performance benchmarking')
@click.pass_context
def evaluate(ctx, model_path, test_data, output_dir, benchmark):
    """Evaluate trained model."""
    click.echo("📊 Starting SwellSight evaluation...")
    
    config_path = ctx.obj.get('config', 'configs/evaluation.yaml')
    click.echo(f"Config: {config_path}")
    click.echo(f"Model: {model_path}")
    click.echo(f"Test data: {test_data}")
    
    # TODO: Implement evaluation in task 11.1
    click.echo("Evaluation pipeline will be implemented in task 11.1")

@cli.command()
@click.option('--input', type=str, required=True, help='Input image or directory')
@click.option('--output', type=str, default='outputs/inference', help='Output directory')
@click.option('--model-path', type=str, help='Model checkpoint path')
@click.option('--batch-size', type=int, default=1, help='Batch size')
@click.pass_context
def analyze(ctx, input, output, model_path, batch_size):
    """Analyze wave conditions from images."""
    click.echo("🔍 Starting SwellSight analysis...")
    
    config_path = ctx.obj.get('config', 'configs/inference.yaml')
    click.echo(f"Config: {config_path}")
    click.echo(f"Input: {input}")
    click.echo(f"Output: {output}")
    
    # TODO: Implement inference in task 13.1
    click.echo("Inference pipeline will be implemented in task 13.1")

@cli.command()
@click.option('--host', type=str, default='0.0.0.0', help='Server host')
@click.option('--port', type=int, default=8000, help='Server port')
@click.option('--workers', type=int, default=1, help='Number of workers')
@click.pass_context
def serve(ctx, host, port, workers):
    """Start API server."""
    click.echo("🚀 Starting SwellSight API server...")
    
    config_path = ctx.obj.get('config', 'configs/inference.yaml')
    click.echo(f"Config: {config_path}")
    click.echo(f"Server: {host}:{port}")
    
    # TODO: Implement API server in task 14.1
    click.echo("API server will be implemented in task 14.1")

@cli.command()
def version():
    """Show version information."""
    from swellsight import __version__
    click.echo(f"SwellSight Wave Analysis System v{__version__}")

def main():
    """Main CLI entry point."""
    cli()

if __name__ == '__main__':
    main()
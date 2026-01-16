#!/usr/bin/env python3
"""
SwellSight Beach Cam Analyzer

A simple command-line tool for analyzing beach cam images to extract wave metrics.

Usage:
    python analyze_beach_cam.py <image_path>
    python analyze_beach_cam.py <image_path> --output ./results
    python analyze_beach_cam.py <image_path> --gpu --save-intermediates
    python analyze_beach_cam.py batch <image_directory>

Examples:
    # Analyze single image
    python analyze_beach_cam.py beach_cam.jpg
    
    # Analyze with GPU and save results
    python analyze_beach_cam.py beach_cam.jpg --gpu --output ./output
    
    # Batch process directory
    python analyze_beach_cam.py batch ./beach_cams --gpu
    
    # Use CPU only
    python analyze_beach_cam.py beach_cam.jpg --no-gpu
"""

import sys
import argparse
import cv2
import numpy as np
from pathlib import Path
import json
import time

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.swellsight.core.pipeline import WaveAnalysisPipeline, PipelineConfig


def print_banner():
    """Print SwellSight banner."""
    banner = """
    ╔═══════════════════════════════════════════════════════════╗
    ║                                                           ║
    ║   🌊 SwellSight Wave Analysis System 🏄‍♂️                  ║
    ║   AI-Powered Wave Metrics for Surfers                    ║
    ║                                                           ║
    ╚═══════════════════════════════════════════════════════════╝
    """
    print(banner)


def print_results(result, image_name="Image"):
    """Print wave analysis results in a formatted way."""
    print("\n" + "="*70)
    print(f"  WAVE ANALYSIS RESULTS - {image_name}")
    print("="*70)
    
    # Wave metrics
    print(f"\n  📏 Wave Height:     {result.wave_metrics.height_meters:.2f}m "
          f"({result.wave_metrics.height_feet:.1f}ft)")
    print(f"  🧭 Direction:       {result.wave_metrics.direction}")
    print(f"  💥 Breaking Type:   {result.wave_metrics.breaking_type}")
    
    # Confidence scores
    print(f"\n  📊 Confidence Scores:")
    print(f"     Overall:         {result.pipeline_confidence:.1%}")
    print(f"     Height:          {result.wave_metrics.height_confidence:.1%}")
    print(f"     Direction:       {result.wave_metrics.direction_confidence:.1%}")
    print(f"     Breaking:        {result.wave_metrics.breaking_confidence:.1%}")
    
    # Performance
    print(f"\n  ⚡ Performance:")
    print(f"     Processing Time: {result.processing_time:.2f}s")
    if result.stage_timings:
        print(f"     Depth Extract:   {result.stage_timings.get('depth_extraction', 0):.2f}s")
        print(f"     Wave Analysis:   {result.stage_timings.get('wave_analysis', 0):.2f}s")
    
    # Warnings and flags
    if result.wave_metrics.extreme_conditions:
        print(f"\n  ⚠️  EXTREME CONDITIONS DETECTED")
    
    if result.warnings:
        print(f"\n  ⚠️  Warnings:")
        for warning in result.warnings:
            print(f"     - {warning}")
    
    # Quality metrics
    if result.depth_quality:
        print(f"\n  🎯 Quality Metrics:")
        print(f"     Depth Quality:   {result.depth_quality.overall_score:.1%}")
        print(f"     Edge Preserve:   {result.depth_quality.edge_preservation:.1%}")
    
    print("="*70 + "\n")


def analyze_single_image(image_path: str, args):
    """Analyze a single beach cam image."""
    print(f"\n📸 Loading image: {image_path}")
    
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print(f"❌ Error: Could not load image from {image_path}")
        return None
    
    # Convert BGR to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    print(f"✓ Image loaded: {image.shape[1]}x{image.shape[0]} pixels")
    
    # Configure pipeline
    config = PipelineConfig(
        use_gpu=args.gpu,
        enable_optimization=True,
        depth_model_size=args.model_size,
        depth_precision=args.precision,
        save_intermediate_results=args.save_intermediates,
        output_directory=args.output,
        confidence_threshold=args.confidence_threshold
    )
    
    # Initialize pipeline
    print("\n🔧 Initializing wave analysis pipeline...")
    try:
        pipeline = WaveAnalysisPipeline(config)
        print("✓ Pipeline initialized successfully")
    except Exception as e:
        print(f"❌ Failed to initialize pipeline: {e}")
        return None
    
    # Check system status
    if args.verbose:
        status = pipeline.get_pipeline_status()
        print(f"\n📊 System Status:")
        print(f"   GPU Available: {status['hardware_status'].get('gpu_available', False)}")
        print(f"   Components Ready: {status['components_initialized']}")
    
    # Process image
    print("\n🌊 Analyzing waves...")
    start_time = time.time()
    
    try:
        result = pipeline.process_beach_cam_image(image)
        analysis_time = time.time() - start_time
        
        print(f"✓ Analysis completed in {analysis_time:.2f}s")
        
        # Print results
        print_results(result, Path(image_path).name)
        
        # Save results if output directory specified
        if args.output:
            output_dir = Path(args.output)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Save JSON results
            result_file = output_dir / f"{Path(image_path).stem}_results.json"
            result.save_to_file(str(result_file))
            print(f"💾 Results saved to: {result_file}")
            
            # Save depth map if available
            if result.enhanced_depth_map and args.save_intermediates:
                depth_file = output_dir / f"{Path(image_path).stem}_depth.npy"
                np.save(depth_file, result.enhanced_depth_map.data)
                print(f"💾 Depth map saved to: {depth_file}")
        
        return result
        
    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return None


def analyze_batch(directory: str, args):
    """Analyze all images in a directory."""
    print(f"\n📁 Scanning directory: {directory}")
    
    # Find all image files
    image_dir = Path(directory)
    image_extensions = ['.jpg', '.jpeg', '.png', '.webp']
    image_files = []
    
    for ext in image_extensions:
        image_files.extend(image_dir.glob(f'*{ext}'))
        image_files.extend(image_dir.glob(f'*{ext.upper()}'))
    
    if not image_files:
        print(f"❌ No images found in {directory}")
        return
    
    print(f"✓ Found {len(image_files)} images")
    
    # Load images
    print("\n📸 Loading images...")
    images = []
    valid_files = []
    
    for img_path in image_files:
        img = cv2.imread(str(img_path))
        if img is not None:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            images.append(img)
            valid_files.append(img_path)
        else:
            print(f"⚠️  Skipped (could not load): {img_path.name}")
    
    print(f"✓ Loaded {len(images)} images successfully")
    
    # Configure pipeline
    config = PipelineConfig(
        use_gpu=args.gpu,
        enable_optimization=True,
        depth_model_size=args.model_size,
        depth_precision=args.precision,
        save_intermediate_results=args.save_intermediates,
        output_directory=args.output
    )
    
    # Initialize pipeline
    print("\n🔧 Initializing wave analysis pipeline...")
    pipeline = WaveAnalysisPipeline(config)
    print("✓ Pipeline initialized successfully")
    
    # Process batch
    print(f"\n🌊 Analyzing {len(images)} images...")
    
    def progress_callback(current, total, result):
        progress = current / total * 100
        print(f"Progress: [{current}/{total}] {progress:.1f}%", end='\r')
    
    start_time = time.time()
    batch_results = pipeline.process_batch(images, progress_callback=progress_callback)
    total_time = time.time() - start_time
    
    print()  # New line after progress
    
    # Print summary
    print("\n" + "="*70)
    print("  BATCH PROCESSING SUMMARY")
    print("="*70)
    print(f"\n  Total Images:     {len(images)}")
    print(f"  Successful:       {len([r for r in batch_results.individual_results if r])}")
    print(f"  Failed:           {len(batch_results.failed_indices)}")
    print(f"  Success Rate:     {batch_results.get_success_rate():.1%}")
    print(f"\n  Total Time:       {total_time:.2f}s")
    print(f"  Average Time:     {batch_results.get_average_processing_time():.2f}s per image")
    print(f"  Throughput:       {batch_results.batch_statistics['throughput_images_per_second']:.2f} images/sec")
    
    if batch_results.batch_statistics.get('average_confidence'):
        print(f"  Avg Confidence:   {batch_results.batch_statistics['average_confidence']:.1%}")
    
    print("="*70)
    
    # Print individual results
    if args.verbose:
        print("\n📊 Individual Results:")
        for i, (result, img_path) in enumerate(zip(batch_results.individual_results, valid_files)):
            if result:
                print(f"\n  {i+1}. {img_path.name}")
                print(f"     Height: {result.wave_metrics.height_meters:.1f}m, "
                      f"Direction: {result.wave_metrics.direction}, "
                      f"Breaking: {result.wave_metrics.breaking_type}")
                print(f"     Confidence: {result.pipeline_confidence:.1%}, "
                      f"Time: {result.processing_time:.2f}s")
    
    # Save batch summary
    if args.output:
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        summary_file = output_dir / "batch_summary.json"
        summary = {
            "total_images": len(images),
            "successful": len([r for r in batch_results.individual_results if r]),
            "failed": len(batch_results.failed_indices),
            "success_rate": batch_results.get_success_rate(),
            "total_time": total_time,
            "average_time": batch_results.get_average_processing_time(),
            "throughput": batch_results.batch_statistics['throughput_images_per_second'],
            "results": [
                {
                    "filename": img_path.name,
                    "height_meters": result.wave_metrics.height_meters if result else None,
                    "direction": result.wave_metrics.direction if result else None,
                    "breaking_type": result.wave_metrics.breaking_type if result else None,
                    "confidence": result.pipeline_confidence if result else None
                }
                for img_path, result in zip(valid_files, batch_results.individual_results)
            ]
        }
        
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\n💾 Batch summary saved to: {summary_file}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='SwellSight Beach Cam Analyzer - AI-powered wave analysis',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    # Positional arguments
    parser.add_argument('command', nargs='?', default='analyze',
                       help='Command: analyze (default) or batch')
    parser.add_argument('path', help='Path to image file or directory')
    
    # Optional arguments
    parser.add_argument('--output', '-o', type=str, default=None,
                       help='Output directory for results')
    parser.add_argument('--gpu', action='store_true', default=True,
                       help='Use GPU acceleration (default: True)')
    parser.add_argument('--no-gpu', dest='gpu', action='store_false',
                       help='Disable GPU, use CPU only')
    parser.add_argument('--model-size', choices=['small', 'base', 'large'],
                       default='large', help='Model size (default: large)')
    parser.add_argument('--precision', choices=['fp16', 'fp32'],
                       default='fp16', help='Model precision (default: fp16)')
    parser.add_argument('--save-intermediates', action='store_true',
                       help='Save intermediate results (depth maps, etc.)')
    parser.add_argument('--confidence-threshold', type=float, default=0.7,
                       help='Minimum confidence threshold (default: 0.7)')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Verbose output')
    
    args = parser.parse_args()
    
    # Print banner
    print_banner()
    
    # Handle commands
    if args.command == 'batch':
        analyze_batch(args.path, args)
    else:
        # If command is actually a path, treat as single image analysis
        if Path(args.command).exists():
            args.path = args.command
        analyze_single_image(args.path, args)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Analysis interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        sys.exit(1)

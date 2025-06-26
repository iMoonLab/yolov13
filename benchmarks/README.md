# YOLOv13 Benchmark Tools

This directory contains specialized benchmark and analysis tools for YOLOv13 models.

## Available Tools

### 1. Architecture Validator (`yolov13_architecture_validator.py`)

Analyzes YOLOv13's specific architectural components and validates their claimed benefits.

**What it does:**
- Finds and analyzes HyperACE modules for efficiency claims
- Measures gradient flow through FullPAD tunnels
- Calculates parameter reduction in DS-blocks vs standard convolutions
- Provides quantitative validation of architectural innovations

**Usage:**
```bash
python yolov13_architecture_validator.py --model yolov13n.pt
```

**Example output:**
- HyperACE modules: 1 found
- FullPAD tunnels: 7 found with gradient measurements
- DS-blocks: 44 found with 3.0x parameter reduction confirmed

### 2. Deployment Analyzer (`yolov13_deployment_analyzer.py`)

Optimizes YOLOv13 deployment for different hardware and use cases.

**What it does:**
- Analyzes memory footprint for different deployment scenarios
- Tests optimal batch sizes for your hardware
- Compares export format performance (PyTorch, ONNX, TensorRT, TorchScript)
- Provides hardware-specific deployment recommendations

**Usage:**
```bash
python yolov13_deployment_analyzer.py --model yolov13n.pt
```

**Example output:**
- Memory usage: 41.7MB GPU for single inference
- Optimal batch size: 4 (72.0 FPS vs 17.4 FPS for batch 1)
- Best export format: ONNX (24.6 FPS vs 22.1 FPS PyTorch)

### 3. Working Benchmark (`yolov13_benchmark_working.py`)

Simple benchmark tool that tests basic YOLOv13 performance metrics.

**What it does:**
- Measures inference speed and FPS
- Counts model parameters and size
- Tests with different input sizes
- Generates basic performance report

**Usage:**
```bash
python yolov13_benchmark_working.py --model yolov13n.pt
```

### 4. Real Image Test (`yolov13_real_image_test.py`)

Tests YOLOv13 detection on actual images to verify it works correctly.

**What it does:**
- Downloads test images from the internet
- Runs object detection and shows results
- Verifies detection accuracy and confidence scores
- Saves annotated images with bounding boxes

**Usage:**
```bash
python yolov13_real_image_test.py --model yolov13n.pt
```

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run any tool:
```bash
cd yolov13/benchmarks
python <tool_name>.py --model path/to/your/model.pt
```

## Why These Tools Are Useful

### For Researchers
- **Validate claims**: Get quantitative evidence for YOLOv13's architectural innovations
- **Compare architectures**: Measure actual parameter efficiency and performance improvements
- **Scientific analysis**: Generate data for papers and presentations

### For Deployment Engineers
- **Avoid failures**: Find optimal batch sizes before deploying to production
- **Save resources**: Choose the most efficient export format for your hardware
- **Optimize performance**: Get specific recommendations for mobile, edge, or server deployment

### For General Users
- **Verify model works**: Test detection on real images before using in applications
- **Understand performance**: See actual FPS and memory usage on your hardware
- **Choose configurations**: Find the best settings for your specific use case

## Tool Outputs

All tools save their results to JSON files for further analysis:
- `yolov13_validation_report.json` - Architecture validation results
- `yolov13_deployment_report.json` - Deployment optimization results  
- `yolov13_benchmark_results.json` - Basic benchmark results

## Requirements

- Python 3.8+
- PyTorch
- Ultralytics YOLOv13
- CUDA (optional, for GPU acceleration)

See `requirements.txt` for complete dependency list.

## Notes

- Tools work with any YOLOv13 model size (n, s, m, l, x)
- GPU acceleration is automatically detected and used when available
- Results may vary based on hardware and system configuration
- Some tools require internet connection for downloading test images 
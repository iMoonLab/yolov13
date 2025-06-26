"""
YOLOv13 Benchmark Suite

A comprehensive benchmarking toolkit for YOLOv13 model evaluation.
Includes performance analysis, architecture detection, and innovation scoring.
"""

__version__ = "1.0.0"
__author__ = "YOLOv13 Contributors"

from pathlib import Path

# Package information
PACKAGE_DIR = Path(__file__).parent
ROOT_DIR = PACKAGE_DIR.parent

# Available benchmark modules
__all__ = [
    "yolov13_benchmark_working",
    "yolov13_benchmark_suite", 
    "yolov13_real_image_test",
    "test_suite"
] 
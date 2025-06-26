#!/usr/bin/env python3
"""
Comprehensive Test Suite for YOLOv13 Benchmark
Tests all functionality before commit
"""

import time
import json
import traceback
import numpy as np
from pathlib import Path
import sys

def print_test_header(test_name):
    """Print test header"""
    print(f"\n{'='*60}")
    print(f"🧪 {test_name}")
    print(f"{'='*60}")

def print_test_result(test_name, passed, details=""):
    """Print test result"""
    status = "✅ PASSED" if passed else "❌ FAILED"
    print(f"{status} | {test_name}")
    if details:
        print(f"    {details}")

def test_1_basic_model_loading():
    """Test 1: Basic YOLOv13 Model Loading"""
    print_test_header("TEST 1: Basic YOLOv13 Model Loading")
    
    try:
        from ultralytics import YOLO
        model = YOLO('yolov13n.pt')
        
        # Verify model properties
        assert model.task == 'detect', f"Expected detect, got {model.task}"
        assert 'DetectionModel' in str(type(model.model)), "Not a detection model"
        
        print_test_result("Model Loading", True, f"Task: {model.task}")
        return True, model
        
    except Exception as e:
        print_test_result("Model Loading", False, str(e))
        return False, None

def test_2_architecture_analysis(model):
    """Test 2: YOLOv13 Architecture Detection"""
    print_test_header("TEST 2: YOLOv13 Architecture Analysis")
    
    try:
        yolov13_modules = {
            'DSC3k2': 0, 'A2C2f': 0, 'ABlock': 0, 'AAttn': 0,
            'DSConv': 0, 'DSBottleneck': 0, 'HyperACE': 0, 'FullPAD_Tunnel': 0
        }
        
        for name, module in model.model.named_modules():
            module_name = module.__class__.__name__
            if module_name in yolov13_modules:
                yolov13_modules[module_name] += 1
        
        total_modules = sum(yolov13_modules.values())
        unique_types = sum(1 for count in yolov13_modules.values() if count > 0)
        
        # Verify YOLOv13-specific modules exist
        assert total_modules > 50, f"Too few YOLOv13 modules: {total_modules}"
        assert unique_types >= 6, f"Too few unique module types: {unique_types}"
        assert yolov13_modules['DSC3k2'] > 0, "No DSC3k2 modules found"
        assert yolov13_modules['A2C2f'] > 0, "No A2C2f modules found"
        
        print_test_result("Architecture Analysis", True, 
                         f"{total_modules} modules, {unique_types} types")
        return True, yolov13_modules
        
    except Exception as e:
        print_test_result("Architecture Analysis", False, str(e))
        return False, None

def test_3_single_image_inference(model):
    """Test 3: Single Image Inference"""
    print_test_header("TEST 3: Single Image Inference")
    
    try:
        # Test with synthetic image
        test_image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
        
        start_time = time.time()
        results = model(test_image, verbose=False)
        inference_time = time.time() - start_time
        
        # Verify results
        assert len(results) == 1, f"Expected 1 result, got {len(results)}"
        assert hasattr(results[0], 'boxes'), "Result missing boxes attribute"
        assert inference_time < 10.0, f"Inference too slow: {inference_time:.3f}s"
        
        print_test_result("Single Image Inference", True, 
                         f"{inference_time*1000:.1f}ms")
        return True
        
    except Exception as e:
        print_test_result("Single Image Inference", False, str(e))
        return False

def test_4_batch_processing(model):
    """Test 4: Batch Processing"""
    print_test_header("TEST 4: Batch Processing")
    
    try:
        batch_sizes = [2, 4, 8]
        batch_results = []
        
        for batch_size in batch_sizes:
            # Create batch
            batch_images = [np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8) 
                          for _ in range(batch_size)]
            
            start_time = time.time()
            results = model(batch_images, verbose=False)
            batch_time = time.time() - start_time
            
            # Verify batch results
            assert len(results) == batch_size, f"Batch size mismatch: {len(results)} != {batch_size}"
            
            avg_time_per_image = batch_time / batch_size
            batch_results.append((batch_size, batch_time, avg_time_per_image))
            
            print(f"    Batch {batch_size}: {batch_time*1000:.1f}ms total, {avg_time_per_image*1000:.1f}ms/image")
        
        print_test_result("Batch Processing", True, f"Tested batches: {batch_sizes}")
        return True, batch_results
        
    except Exception as e:
        print_test_result("Batch Processing", False, str(e))
        return False, None

def test_5_real_image_detection(model):
    """Test 5: Real Image Detection"""
    print_test_header("TEST 5: Real Image Detection")
    
    try:
        # Test with downloaded real images
        test_images = ['test_bus.jpg', 'test_person.jpg']
        detection_results = []
        
        for img_path in test_images:
            if not Path(img_path).exists():
                print(f"    ⚠️  Skipping {img_path} - file not found")
                continue
                
            results = model(img_path, verbose=False)
            detections = len(results[0].boxes) if results[0].boxes is not None else 0
            detection_results.append((img_path, detections))
            print(f"    {img_path}: {detections} objects detected")
        
        # Verify meaningful detections
        total_detections = sum(det[1] for det in detection_results)
        assert total_detections > 0, "No objects detected in any image"
        
        print_test_result("Real Image Detection", True, 
                         f"{total_detections} total objects detected")
        return True, detection_results
        
    except Exception as e:
        print_test_result("Real Image Detection", False, str(e))
        return False, None

def test_6_benchmark_script_execution():
    """Test 6: Benchmark Script Execution"""
    print_test_header("TEST 6: Benchmark Script Execution")
    
    try:
        import subprocess
        
        # Test the working benchmark script
        result = subprocess.run([
            sys.executable, 'yolov13_benchmark_working.py', 
            '--model', 'yolov13n.pt', '--quick'
        ], capture_output=True, text=True, timeout=60)
        
        # Check if script ran successfully
        success = result.returncode == 0
        
        if success:
            # Check if output file was created
            output_exists = Path('yolov13_benchmark_results.json').exists()
            if output_exists:
                with open('yolov13_benchmark_results.json', 'r') as f:
                    data = json.load(f)
                innovation_score = data.get('innovation_score', 0)
                print_test_result("Benchmark Script", True, 
                                f"Innovation score: {innovation_score}")
            else:
                print_test_result("Benchmark Script", False, "No output file created")
                success = False
        else:
            print_test_result("Benchmark Script", False, f"Exit code: {result.returncode}")
            
        return success
        
    except subprocess.TimeoutExpired:
        print_test_result("Benchmark Script", False, "Script timeout")
        return False
    except Exception as e:
        print_test_result("Benchmark Script", False, str(e))
        return False

def test_7_real_image_script_execution():
    """Test 7: Real Image Test Script"""
    print_test_header("TEST 7: Real Image Test Script")
    
    try:
        import subprocess
        
        result = subprocess.run([
            sys.executable, 'yolov13_real_image_test.py', 
            '--model', 'yolov13n.pt', '--runs', '2'
        ], capture_output=True, text=True, timeout=90)
        
        success = result.returncode == 0
        
        if success:
            # Check for output file
            output_files = list(Path('.').glob('real_image_test_results_*.json'))
            if output_files:
                print_test_result("Real Image Script", True, 
                                f"Output: {output_files[-1].name}")
            else:
                print_test_result("Real Image Script", False, "No output file found")
                success = False
        else:
            print_test_result("Real Image Script", False, f"Exit code: {result.returncode}")
            
        return success
        
    except subprocess.TimeoutExpired:
        print_test_result("Real Image Script", False, "Script timeout")
        return False
    except Exception as e:
        print_test_result("Real Image Script", False, str(e))
        return False

def test_8_performance_validation(model):
    """Test 8: Performance Validation"""
    print_test_header("TEST 8: Performance Validation")
    
    try:
        # Test inference speed consistency
        test_image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
        times = []
        
        # Warmup
        for _ in range(3):
            model(test_image, verbose=False)
        
        # Measure performance
        for _ in range(10):
            start_time = time.time()
            model(test_image, verbose=False)
            times.append(time.time() - start_time)
        
        avg_time = sum(times) / len(times)
        fps = 1.0 / avg_time
        std_dev = np.std(times)
        
        # Performance checks
        assert avg_time < 0.5, f"Average inference too slow: {avg_time:.3f}s"
        assert fps > 5, f"FPS too low: {fps:.1f}"
        assert std_dev < 0.1, f"Timing too inconsistent: {std_dev:.3f}s std dev"
        
        print_test_result("Performance Validation", True, 
                         f"{fps:.1f} FPS, {std_dev*1000:.1f}ms std dev")
        return True
        
    except Exception as e:
        print_test_result("Performance Validation", False, str(e))
        return False

def main():
    """Run comprehensive test suite"""
    print("🚀 YOLOv13 Benchmark Suite - Comprehensive Testing")
    print("🎯 Ensuring production readiness before commit")
    
    start_time = time.time()
    test_results = []
    model = None
    
    # Run all tests
    tests = [
        ("Basic Model Loading", test_1_basic_model_loading),
        ("Architecture Analysis", lambda: test_2_architecture_analysis(model)),
        ("Single Image Inference", lambda: test_3_single_image_inference(model)),
        ("Batch Processing", lambda: test_4_batch_processing(model)),
        ("Real Image Detection", lambda: test_5_real_image_detection(model)),
        ("Benchmark Script", test_6_benchmark_script_execution),
        ("Real Image Script", test_7_real_image_script_execution),
        ("Performance Validation", lambda: test_8_performance_validation(model))
    ]
    
    for test_name, test_func in tests:
        try:
            if test_name == "Basic Model Loading":
                success, model = test_func()
            else:
                success = test_func()
                if isinstance(success, tuple):
                    success = success[0]
            
            test_results.append((test_name, success))
            
        except Exception as e:
            print(f"❌ CRITICAL ERROR in {test_name}: {e}")
            traceback.print_exc()
            test_results.append((test_name, False))
    
    # Summary
    total_time = time.time() - start_time
    passed_tests = sum(1 for _, success in test_results if success)
    total_tests = len(test_results)
    
    print(f"\n{'='*60}")
    print("📊 TEST SUMMARY")
    print(f"{'='*60}")
    
    for test_name, success in test_results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} | {test_name}")
    
    print(f"\n🎯 Results: {passed_tests}/{total_tests} tests passed")
    print(f"⏱️  Total time: {total_time:.1f}s")
    
    if passed_tests == total_tests:
        print("🎉 ALL TESTS PASSED - READY FOR COMMIT! 🚀")
        return 0
    else:
        print(f"❌ {total_tests - passed_tests} tests failed - Fix before commit")
        return 1

if __name__ == '__main__':
    sys.exit(main()) 
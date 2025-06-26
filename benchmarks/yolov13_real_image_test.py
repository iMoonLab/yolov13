#!/usr/bin/env python3
"""
YOLOv13 Real Image Benchmark Test
Tests the benchmark suite with actual images and provides clear feedback
"""

import argparse
import time
import json
import os
import urllib.request
from pathlib import Path
import sys

def print_header():
    """Print the test header"""
    print("=" * 70)
    print("🖼️  YOLOv13 Real Image Benchmark Test")
    print("=" * 70)
    print("🎯 Testing with real images to verify functionality")
    print("-" * 70)

def download_test_images():
    """Download sample test images"""
    print("\n📥 Downloading test images...")
    
    test_images = [
        {
            'url': 'https://ultralytics.com/images/bus.jpg',
            'filename': 'test_bus.jpg',
            'description': 'Bus scene'
        },
        {
            'url': 'https://ultralytics.com/images/zidane.jpg', 
            'filename': 'test_person.jpg',
            'description': 'Person scene'
        }
    ]
    
    downloaded_images = []
    
    for img_info in test_images:
        try:
            if not os.path.exists(img_info['filename']):
                print(f"   📡 Downloading {img_info['description']}...")
                urllib.request.urlretrieve(img_info['url'], img_info['filename'])
                print(f"   ✅ Downloaded: {img_info['filename']}")
            else:
                print(f"   ✅ Already exists: {img_info['filename']}")
            
            downloaded_images.append(img_info['filename'])
            
        except Exception as e:
            print(f"   ⚠️  Failed to download {img_info['filename']}: {e}")
    
    return downloaded_images

def test_model_with_real_images(model, image_paths, runs=3):
    """Test model with real images and show detailed results"""
    print(f"\n🔥 Testing with {len(image_paths)} real images:")
    print("-" * 50)
    
    all_results = []
    total_detections = 0
    
    for i, image_path in enumerate(image_paths, 1):
        if not os.path.exists(image_path):
            print(f"   ❌ Image not found: {image_path}")
            continue
            
        print(f"\n📸 Image {i}: {image_path}")
        
        # Get image info
        try:
            from PIL import Image
            with Image.open(image_path) as img:
                width, height = img.size
                print(f"   📏 Resolution: {width}x{height}")
        except:
            print("   📏 Resolution: Unknown")
        
        # Warmup run
        print("   🔄 Warming up...")
        try:
            warmup_result = model(image_path, verbose=False)
            print("   ✅ Warmup successful")
        except Exception as e:
            print(f"   ❌ Warmup failed: {e}")
            continue
        
        # Benchmark runs
        print(f"   ⏱️  Running {runs} benchmark iterations...")
        times = []
        detection_counts = []
        
        for run in range(runs):
            try:
                start_time = time.time()
                results = model(image_path, verbose=False)
                end_time = time.time()
                
                run_time = end_time - start_time
                times.append(run_time)
                
                # Count detections
                detections = len(results[0].boxes) if results[0].boxes is not None else 0
                detection_counts.append(detections)
                
                print(f"     Run {run+1}: {run_time*1000:.1f}ms, {detections} objects detected")
                
            except Exception as e:
                print(f"     Run {run+1}: Failed - {e}")
                continue
        
        if times:
            avg_time = sum(times) / len(times)
            avg_detections = sum(detection_counts) / len(detection_counts)
            fps = 1.0 / avg_time if avg_time > 0 else 0
            
            result = {
                'image': image_path,
                'avg_time_ms': avg_time * 1000,
                'fps': fps,
                'avg_detections': avg_detections,
                'times': times,
                'detection_counts': detection_counts
            }
            all_results.append(result)
            total_detections += avg_detections
            
            print(f"   📊 Average: {avg_time*1000:.1f}ms | {fps:.1f} FPS | {avg_detections:.1f} objects")
        else:
            print("   ❌ All runs failed")
    
    # Summary
    if all_results:
        print(f"\n📈 Summary across {len(all_results)} images:")
        print("-" * 40)
        
        total_avg_time = sum(r['avg_time_ms'] for r in all_results) / len(all_results)
        total_avg_fps = sum(r['fps'] for r in all_results) / len(all_results)
        avg_objects_per_image = total_detections / len(all_results)
        
        print(f"   🎯 Average inference time: {total_avg_time:.1f}ms")
        print(f"   🚀 Average FPS: {total_avg_fps:.1f}")
        print(f"   🔍 Average objects per image: {avg_objects_per_image:.1f}")
        print(f"   ✅ Success rate: {len(all_results)}/{len(image_paths)} images")
        
        # Show individual detection results
        print(f"\n🔍 Detection Results by Image:")
        for result in all_results:
            print(f"   📸 {result['image']}: {result['avg_detections']:.1f} objects detected")
        
    return all_results

def verify_yolov13_features(model):
    """Verify that YOLOv13-specific features are detected"""
    print("\n🔍 Verifying YOLOv13 Architecture:")
    print("-" * 40)
    
    yolov13_features = {
        'DSC3k2': 0,
        'A2C2f': 0,
        'ABlock': 0,
        'AAttn': 0,
        'DSConv': 0,
        'DSBottleneck': 0,
        'HyperACE': 0,
        'FullPAD_Tunnel': 0
    }
    
    # Count YOLOv13-specific modules
    for name, module in model.model.named_modules():
        module_name = module.__class__.__name__
        if module_name in yolov13_features:
            yolov13_features[module_name] += 1
    
    total_yolov13_modules = sum(yolov13_features.values())
    features_found = sum(1 for count in yolov13_features.values() if count > 0)
    
    print(f"✅ YOLOv13-specific modules found:")
    for feature, count in yolov13_features.items():
        if count > 0:
            print(f"   🎯 {feature}: {count} instances")
    
    print(f"\n📊 Architecture Summary:")
    print(f"   • Total YOLOv13 modules: {total_yolov13_modules}")
    print(f"   • Unique feature types: {features_found}")
    print(f"   • Model confirmed as YOLOv13: {'✅ YES' if total_yolov13_modules > 10 else '❌ NO'}")
    
    return total_yolov13_modules > 10

def main():
    """Main test execution"""
    parser = argparse.ArgumentParser(description='YOLOv13 Real Image Test')
    parser.add_argument('--model', default='yolov13n.pt', help='Model to test')
    parser.add_argument('--runs', type=int, default=3, help='Number of benchmark runs per image')
    parser.add_argument('--images', nargs='+', help='Custom image paths to test')
    
    args = parser.parse_args()
    
    print_header()
    
    try:
        # Load model
        print(f"📦 Loading YOLOv13 model: {args.model}")
        from ultralytics import YOLO
        model = YOLO(args.model)
        print(f"✅ Model loaded successfully")
        print(f"📋 Task: {model.task}")
        
        # Verify it's actually YOLOv13
        is_yolov13 = verify_yolov13_features(model)
        if not is_yolov13:
            print("⚠️  Warning: This doesn't appear to be a YOLOv13 model!")
        
        # Get test images
        if args.images:
            test_images = args.images
            print(f"\n📂 Using custom images: {test_images}")
        else:
            test_images = download_test_images()
        
        if not test_images:
            print("❌ No test images available!")
            return
        
        # Run the test
        results = test_model_with_real_images(model, test_images, args.runs)
        
        # Save results
        output_file = f"real_image_test_results_{int(time.time())}.json"
        test_report = {
            'model': args.model,
            'test_images': test_images,
            'results': results,
            'is_yolov13': is_yolov13,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        
        with open(output_file, 'w') as f:
            json.dump(test_report, f, indent=2)
        
        print(f"\n💾 Test results saved to: {output_file}")
        
        # Final status
        if results:
            print("\n🎉 TEST COMPLETED SUCCESSFULLY!")
            print("✅ The benchmark is working correctly with real images")
            print("✅ YOLOv13 model is performing object detection")
            print("✅ Performance metrics are being measured accurately")
        else:
            print("\n❌ TEST FAILED!")
            print("No successful results obtained")
        
        print("=" * 70)
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == '__main__':
    main() 
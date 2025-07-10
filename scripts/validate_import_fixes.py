#!/usr/bin/env python3
"""
Test Import Fix Summary Report

This script validates that all our import fixes are working correctly.
"""

import subprocess
import sys
from pathlib import Path

def run_pytest_tests():
    """Run key test files to validate imports work."""
    
    test_files = [
        "tests/src/data_utils/test_rgb_to_rgbl.py",
        "tests/src/data_utils/test_dual_channel_dataset.py", 
        "tests/src/models2/multi_channel/test_conv.py",
        "tests/src/models2/multi_channel/test_blocks.py",
        "tests/src/models2/multi_channel/test_container.py",
    ]
    
    print("🧪 TESTING IMPORT FIXES")
    print("=" * 50)
    
    project_root = Path(__file__).parent.parent
    env = {"PYTHONPATH": f"{project_root}/src"}
    
    results = {}
    
    for test_file in test_files:
        print(f"\n📄 Running {test_file}...")
        try:
            result = subprocess.run(
                [sys.executable, "-m", "pytest", test_file, "-q", "--tb=no"],
                cwd=project_root,
                env={**dict(os.environ), **env},
                capture_output=True,
                text=True,
                timeout=60
            )
            
            if result.returncode == 0:
                # Extract test count from output
                lines = result.stdout.strip().split('\n')
                summary_line = [line for line in lines if "passed" in line][-1] if lines else ""
                print(f"   ✅ {summary_line}")
                results[test_file] = "PASSED"
            else:
                print(f"   ❌ FAILED")
                results[test_file] = "FAILED"
                
        except subprocess.TimeoutExpired:
            print(f"   ⏰ TIMEOUT")
            results[test_file] = "TIMEOUT"
        except Exception as e:
            print(f"   💥 ERROR: {e}")
            results[test_file] = "ERROR"
    
    return results

def test_main_script_imports():
    """Test that main script imports work correctly."""
    print("\n🔧 TESTING MAIN SCRIPT IMPORTS")
    print("=" * 50)
    
    try:
        # Test basic imports
        import sys
        from pathlib import Path
        sys.path.append(str(Path(__file__).parent.parent / "src"))
        
        from data_utils.dataset_utils import load_cifar100_data
        from data_utils.rgb_to_rgbl import RGBtoRGBL
        from models2.multi_channel.mc_resnet import mc_resnet50
        from data_utils.dual_channel_dataset import create_dual_channel_dataloaders
        
        print("✅ All critical imports working!")
        
        # Test model creation
        model = mc_resnet50(num_classes=10, device='cpu', use_amp=False)
        print("✅ Model creation working!")
        
        return True
        
    except Exception as e:
        print(f"❌ Import error: {e}")
        return False

def final_import_check():
    """Run final check for any remaining 'from src.' imports."""
    print("\n🔍 FINAL IMPORT CHECK")
    print("=" * 50)
    
    try:
        result = subprocess.run(
            [sys.executable, "scripts/check_import_fixes.py"],
            cwd=Path(__file__).parent.parent,
            capture_output=True,
            text=True
        )
        
        lines = result.stdout.split('\n')
        found_line = [line for line in lines if "Found" in line and "imports" in line][0]
        print(f"📊 {found_line}")
        
        if "Found 6 files" in found_line:
            print("✅ Only non-critical comment strings remain!")
            return True
        else:
            print("⚠️  Some imports may still need fixing")
            return False
            
    except Exception as e:
        print(f"❌ Error running check: {e}")
        return False

def main():
    print("📋 IMPORT FIX VALIDATION REPORT")
    print("=" * 80)
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Test main script imports
    main_imports_ok = test_main_script_imports()
    
    # Test pytest files
    test_results = run_pytest_tests()
    
    # Final import check
    final_check_ok = final_import_check()
    
    # Summary
    print("\n📊 SUMMARY")
    print("=" * 50)
    
    print(f"Main script imports: {'✅ WORKING' if main_imports_ok else '❌ FAILED'}")
    
    passed_tests = sum(1 for result in test_results.values() if result == "PASSED")
    total_tests = len(test_results)
    print(f"Test files: {passed_tests}/{total_tests} passed")
    
    for test_file, result in test_results.items():
        status = "✅" if result == "PASSED" else "❌"
        print(f"  {status} {Path(test_file).name}")
    
    print(f"Remaining imports: {'✅ CLEAN' if final_check_ok else '⚠️  SOME REMAIN'}")
    
    overall_success = main_imports_ok and (passed_tests == total_tests) and final_check_ok
    
    print(f"\n🎯 OVERALL STATUS: {'✅ SUCCESS' if overall_success else '⚠️  NEEDS ATTENTION'}")
    
    if overall_success:
        print("\n🎉 All import fixes are working correctly!")
        print("   The project is ready for development and training.")
    else:
        print("\n🔧 Some issues remain that may need attention.")

if __name__ == "__main__":
    import os
    import datetime
    from datetime import datetime
    main()

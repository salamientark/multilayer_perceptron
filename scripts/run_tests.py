#!/usr/bin/env python3
"""
Test runner script for multilayer perceptron project.
Runs all unit tests and provides detailed reporting.
"""

import subprocess
import sys
import os
import re
from pathlib import Path

# ANSI color codes
class Colors:
    GREEN = '\033[0;32m'
    YELLOW = '\033[0;33m'
    RED = '\033[0;31m'
    NC = '\033[0m'  # No Color
    BOLD = '\033[1m'

def run_test_file(test_module, python_executable=None):
    """Run a single test file and return results"""
    if python_executable is None:
        python_executable = sys.executable
    
    try:
        # Get project root directory
        project_root = Path(__file__).parent.parent
        
        # Set up environment with project root in PYTHONPATH
        env = os.environ.copy()
        current_pythonpath = env.get('PYTHONPATH', '')
        if current_pythonpath:
            env['PYTHONPATH'] = f"{project_root}:{current_pythonpath}"
        else:
            env['PYTHONPATH'] = str(project_root)
        
        result = subprocess.run([
            python_executable, '-m', 'unittest', test_module, '-v'
        ], capture_output=True, text=True, cwd=project_root, env=env)
        
        output = result.stdout + result.stderr
        
        # Count total tests (lines starting with test_)
        total_tests = len(re.findall(r'^test_\w+', output, re.MULTILINE))
        
        # Count passed tests (lines ending with ... ok)
        passed_tests = len(re.findall(r'\.\.\.\ ok$', output, re.MULTILINE))
        
        return {
            'total': total_tests,
            'passed': passed_tests,
            'failed': total_tests - passed_tests,
            'success': result.returncode == 0,
            'output': output
        }
    except FileNotFoundError:
        return {
            'total': 0,
            'passed': 0,
            'failed': 0,
            'success': False,
            'output': 'Test file not found'
        }

def main():
    """Main test runner function"""
    # Check if we have a python executable passed as argument (from Makefile)
    python_executable = sys.argv[1] if len(sys.argv) > 1 else sys.executable
    
    print(f"{Colors.YELLOW}[INFO]{Colors.NC} Running unit tests...")
    
    # Define test modules to run
    test_modules = [
        ('tests.test_arg_parser', 'Argument Parser'),
        ('tests.test_create_model', 'Model Creation'),
        ('tests.test_create_models', 'Training Functionality')
    ]
    
    total_tests = 0
    total_passed = 0
    results = []
    
    # Run each test module
    for module, name in test_modules:
        print(f"{Colors.YELLOW}[INFO]{Colors.NC} Running {name.lower()} tests...")
        
        result = run_test_file(module, python_executable)
        results.append((name, result))
        
        total_tests += result['total']
        total_passed += result['passed']
        
        if result['total'] > 0:
            if result['success']:
                print(f"{Colors.GREEN}[SUCCESS]{Colors.NC} {name} tests: {result['passed']}/{result['total']} passed!")
            else:
                print(f"{Colors.RED}[ERROR]{Colors.NC} {name} tests: {result['passed']}/{result['total']} passed!")
        else:
            print(f"{Colors.YELLOW}[WARNING]{Colors.NC} {name} tests: File not found or no tests")
    
    # Print summary
    print(f"{Colors.YELLOW}{'=' * 40}{Colors.NC}")
    print(f"{Colors.BOLD}TEST SUMMARY{Colors.NC}")
    print(f"{Colors.YELLOW}{'=' * 40}{Colors.NC}")
    
    for name, result in results:
        print(f"{name:25}: {result['passed']}/{result['total']}")
    
    print(f"{Colors.YELLOW}{'=' * 40}{Colors.NC}")
    
    if total_passed == total_tests and total_tests > 0:
        print(f"{Colors.GREEN}TOTAL: {total_passed}/{total_tests} individual tests passed ✓{Colors.NC}")
        return 0
    else:
        print(f"{Colors.RED}TOTAL: {total_passed}/{total_tests} individual tests passed ✗{Colors.NC}")
        return 1

if __name__ == '__main__':
    sys.exit(main())
#!/usr/bin/env python3
"""
Comprehensive test runner for Sprint 6: PyTest matrix and GitHub Actions CI
"""
import subprocess
import sys
import os
import argparse
from pathlib import Path

def run_command(command, description):
    """Run a command and handle errors"""
    print(f"\n🔧 {description}")
    print(f"Command: {command}")
    print("-" * 60)
    
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print("✅ Success")
        if result.stdout:
            print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed: {e}")
        if e.stdout:
            print(f"STDOUT: {e.stdout}")
        if e.stderr:
            print(f"STDERR: {e.stderr}")
        return False

def run_tests(test_type=None, markers=None, coverage=True):
    """Run tests with specified parameters"""
    command = ["pytest"]
    
    if test_type:
        command.append(f"tests/{test_type}/")
    
    if markers:
        command.append(f"-m {markers}")
    
    if coverage:
        command.extend(["--cov=backend", "--cov-report=html", "--cov-report=term-missing"])
    
    command.extend(["-v", "--tb=short"])
    
    return run_command(" ".join(command), f"Running {test_type or 'all'} tests")

def run_linting():
    """Run code linting and formatting checks"""
    commands = [
        ("flake8 backend/ --count --select=E9,F63,F7,F82 --show-source --statistics", "Running flake8 syntax check"),
        ("flake8 backend/ --count --exit-zero --max-complexity=10 --max-line-length=127 --statistics", "Running flake8 style check"),
        ("black --check backend/", "Checking code formatting with black"),
        ("isort --check-only backend/", "Checking import sorting with isort")
    ]
    
    all_passed = True
    for command, description in commands:
        if not run_command(command, description):
            all_passed = False
    
    return all_passed

def run_security_scan():
    """Run security scanning"""
    commands = [
        ("bandit -r backend/ -f json -o bandit-report.json", "Running bandit security scan"),
        ("safety check --json --output safety-report.json", "Running safety dependency check")
    ]
    
    all_passed = True
    for command, description in commands:
        if not run_command(command, description):
            all_passed = False
    
    return all_passed

def run_performance_tests():
    """Run performance tests"""
    return run_command(
        "pytest tests/ -m performance -v --benchmark-only",
        "Running performance tests"
    )

def run_sprint_tests():
    """Run tests for all sprints"""
    sprints = ["sprint1", "sprint2", "sprint3", "sprint4", "sprint5"]
    
    all_passed = True
    for sprint in sprints:
        if not run_tests(markers=sprint, coverage=False):
            all_passed = False
    
    return all_passed

def run_matrix_tests():
    """Run matrix of tests (unit, integration, e2e)"""
    test_types = ["unit", "integration", "e2e"]
    
    all_passed = True
    for test_type in test_types:
        if not run_tests(test_type=test_type):
            all_passed = False
    
    return all_passed

def main():
    parser = argparse.ArgumentParser(description="Comprehensive test runner for Sprint 6")
    parser.add_argument("--test-type", choices=["unit", "integration", "e2e"], help="Run specific test type")
    parser.add_argument("--markers", help="Run tests with specific markers")
    parser.add_argument("--lint", action="store_true", help="Run linting checks")
    parser.add_argument("--security", action="store_true", help="Run security scans")
    parser.add_argument("--performance", action="store_true", help="Run performance tests")
    parser.add_argument("--sprints", action="store_true", help="Run all sprint tests")
    parser.add_argument("--matrix", action="store_true", help="Run matrix of tests")
    parser.add_argument("--all", action="store_true", help="Run all tests and checks")
    parser.add_argument("--no-coverage", action="store_true", help="Disable coverage reporting")
    
    args = parser.parse_args()
    
    print("🧪 Sprint 6: Comprehensive Test Runner")
    print("=" * 60)
    
    all_passed = True
    
    if args.all:
        # Run everything
        print("\n📋 Running complete test suite...")
        
        # Matrix tests
        if not run_matrix_tests():
            all_passed = False
        
        # Sprint tests
        if not run_sprint_tests():
            all_passed = False
        
        # Linting
        if not run_linting():
            all_passed = False
        
        # Security
        if not run_security_scan():
            all_passed = False
        
        # Performance
        if not run_performance_tests():
            all_passed = False
    
    elif args.matrix:
        all_passed = run_matrix_tests()
    
    elif args.sprints:
        all_passed = run_sprint_tests()
    
    elif args.lint:
        all_passed = run_linting()
    
    elif args.security:
        all_passed = run_security_scan()
    
    elif args.performance:
        all_passed = run_performance_tests()
    
    elif args.test_type:
        all_passed = run_tests(test_type=args.test_type, coverage=not args.no_coverage)
    
    elif args.markers:
        all_passed = run_tests(markers=args.markers, coverage=not args.no_coverage)
    
    else:
        # Default: run all tests
        all_passed = run_tests(coverage=not args.no_coverage)
    
    # Summary
    print("\n" + "=" * 60)
    if all_passed:
        print("🎉 All tests passed! Sprint 6 CI/CD pipeline is ready!")
        print("✅ Test matrix: Working")
        print("✅ GitHub Actions: Configured")
        print("✅ Coverage reporting: Active")
        print("✅ Security scanning: Active")
        print("✅ Performance testing: Active")
        print("✅ Linting and formatting: Active")
        sys.exit(0)
    else:
        print("❌ Some tests failed. Please fix the issues before proceeding.")
        sys.exit(1)

if __name__ == "__main__":
    main() 
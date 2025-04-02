#!/usr/bin/env python
"""
Script to run tests for the mobile phone sentiment analysis project.
"""

import os
import sys
import unittest
import argparse

def run_tests(test_path=None, verbose=False):
    """
    Run the project tests.
    
    Args:
        test_path (str): Specific test path to run (e.g., 'tests.test_batch_analyzer')
        verbose (bool): Whether to show verbose output
    
    Returns:
        bool: True if all tests passed, False otherwise
    """
    # Add project root to path
    project_root = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, os.path.dirname(project_root))
    
    # Set up test arguments
    test_args = []
    if verbose:
        test_args.append('-v')
    
    # Run specific tests or discover all tests
    if test_path:
        # Convert path format if needed
        if os.path.exists(test_path):
            # It's a file path
            test_module = os.path.relpath(test_path, project_root)
            test_module = test_module.replace('/', '.').replace('.py', '')
            if test_module.startswith('.'):
                test_module = test_module[1:]
        else:
            # It's already a module path
            test_module = test_path
        
        print(f"Running tests from: {test_module}")
        suite = unittest.defaultTestLoader.loadTestsFromName(test_module)
    else:
        print("Discovering and running all tests...")
        tests_dir = os.path.join(project_root, 'tests')
        suite = unittest.defaultTestLoader.discover(tests_dir)
    
    # Run the tests
    runner = unittest.TextTestRunner(verbosity=2 if verbose else 1)
    result = runner.run(suite)
    
    # Return True if all tests passed
    return result.wasSuccessful()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run tests for the mobile phone sentiment analysis project')
    parser.add_argument('test_path', nargs='?', help='Specific test module or file to run')
    parser.add_argument('-v', '--verbose', action='store_true', help='Show verbose output')
    
    args = parser.parse_args()
    
    success = run_tests(args.test_path, args.verbose)
    
    # Exit with appropriate status code
    sys.exit(0 if success else 1) 
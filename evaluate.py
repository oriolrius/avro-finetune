#!/usr/bin/env python3
"""
Simple evaluation script with default configuration.
For advanced evaluation options, use evaluate_configurable.py
"""

import sys
from evaluate_configurable import main

if __name__ == "__main__":
    print("Running evaluation with default configuration...")
    print("For advanced options, use: uv run python evaluate_configurable.py --help")
    print("-" * 50)
    
    # Run with default arguments (evaluates ./avro-phi3-adapters)
    sys.argv = [sys.argv[0]]  # Keep only script name, remove any arguments
    main()
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Calculate average Success rate from all log files starting with eval_env
"""

import os
import re
import argparse
from pathlib import Path

def extract_success_rate(file_path):
    """
    Extract Success rate value from log file
    
    Args:
        file_path: Path to the log file
        
    Returns:
        float: Success rate value, returns None if not found
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        # Use regex to match "Success rate: X.XX" format
        pattern = r'Success rate:\s*([\d.]+)'
        match = re.search(pattern, content)
        
        if match:
            return float(match.group(1))
        else:
            print(f"Warning: Success rate not found in file {file_path}")
            return None
    except Exception as e:
        print(f"Error: Failed to read file {file_path}: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description="Calculate average Success rate from eval_env*.log files.")
    parser.add_argument(
        "--log_dir",
        type=str,
        default="./eval_logs",
        help="Directory containing eval_env*.log files",
    )
    args = parser.parse_args()

    eval_dir = Path(args.log_dir)
    if not eval_dir.exists():
        print(f"Error: log_dir does not exist: {eval_dir}")
        return
    if not eval_dir.is_dir():
        print(f"Error: log_dir is not a directory: {eval_dir}")
        return
    
    # Get all .log files starting with eval_env
    eval_files = sorted([f for f in eval_dir.glob('eval_env*.log')])
    
    if not eval_files:
        print(f"Error: No log files starting with eval_env found in directory {eval_dir}")
        return
    
    print(f"Found {len(eval_files)} evaluation log files\n")
    
    success_rates = []
    task_results = []
    
    # Iterate through all files and extract Success rate
    for file_path in eval_files:
        success_rate = extract_success_rate(file_path)
        if success_rate is not None:
            success_rates.append(success_rate)
            task_name = file_path.stem.replace('eval_env_', '')
            task_results.append((task_name, success_rate))
            print(f"{task_name}: {success_rate:.2f}")
    
    print(f"\n{'='*60}")
    print(f"Total processed {len(success_rates)} tasks")
    
    if success_rates:
        avg_success_rate = sum(success_rates) / len(success_rates)
        print(f"Average Success rate: {avg_success_rate:.4f} ({avg_success_rate*100:.2f}%)")
        print(f"Highest Success rate: {max(success_rates):.4f} ({max(success_rates)*100:.2f}%)")
        print(f"Lowest Success rate: {min(success_rates):.4f} ({min(success_rates)*100:.2f}%)")
    else:
        print("Error: Failed to extract Success rate from any file")
    
    # Save detailed results to file
    output_file = eval_dir / 'success_rate_summary.txt'
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("Task evaluation results summary\n")
        f.write("="*60 + "\n\n")
        for task_name, rate in task_results:
            f.write(f"{task_name}: {rate:.4f} ({rate*100:.2f}%)\n")
        f.write("\n" + "="*60 + "\n")
        if success_rates:
            f.write(f"Average Success rate: {avg_success_rate:.4f} ({avg_success_rate*100:.2f}%)\n")
            f.write(f"Highest Success rate: {max(success_rates):.4f} ({max(success_rates)*100:.2f}%)\n")
            f.write(f"Lowest Success rate: {min(success_rates):.4f} ({min(success_rates)*100:.2f}%)\n")
    
    print(f"\nDetailed results saved to: {output_file}")

if __name__ == '__main__':
    main()


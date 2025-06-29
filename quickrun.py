#!/usr/bin/env python3

"""
Quick run script for the Bitcoin Policy Ablation Study
This script simplifies running the analysis by providing quick options
"""

import sys
import os
import subprocess

def main():
    """Main execution function"""
    print("Bitcoin Policy Ablation Study")
    print("============================")
    print("1) Run full analysis")
    print("2) Run data preparation only")
    print("3) Run model training only")
    print("4) Run scenario generation only")
    print("5) Run visualization only")
    print("q) Quit")
    
    choice = input("\nEnter your choice: ").strip().lower()
    
    python_cmd = "python" if os.system("python --version") == 0 else "py"
    
    if choice == "1":
        subprocess.run(f"{python_cmd} btc.py", shell=True)
    elif choice == "2":
        subprocess.run(f"{python_cmd} run_btc_analysis.py --data", shell=True)
    elif choice == "3":
        subprocess.run(f"{python_cmd} run_btc_analysis.py --train", shell=True)
    elif choice == "4":
        subprocess.run(f"{python_cmd} run_btc_analysis.py --scenarios", shell=True)
    elif choice == "5":
        subprocess.run(f"{python_cmd} run_btc_analysis.py --visualize", shell=True)
    elif choice == "q":
        print("Exiting program.")
        return 0
    else:
        print("Invalid choice. Please try again.")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())

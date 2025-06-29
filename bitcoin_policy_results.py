"""
Bitcoin Policy Analysis: Complete Results and Findings
This script provides easy access to all visualizations and findings from the Bitcoin policy analysis.
"""
import os
import webbrowser
import time
import sys

def clear_screen():
    """Clear the console screen"""
    os.system('cls' if os.name == 'nt' else 'clear')

def print_header():
    """Print a nice header"""
    print("\n" + "="*80)
    print("                  BITCOIN POLICY ABLATION STUDY - COMPLETE RESULTS                ")
    print("="*80 + "\n")

def print_section(title):
    """Print a section header"""
    print("\n" + "-"*80)
    print(f"  {title}")
    print("-"*80)

def display_menu_and_get_choice(options):
    """Display menu and get user choice"""
    for i, option in enumerate(options, 1):
        print(f"  {i}. {option}")
    
    print("\n  0. Exit")
    
    while True:
        try:
            choice = input("\nEnter your choice (number): ")
            choice = int(choice)
            if 0 <= choice <= len(options):
                return choice
            else:
                print("Invalid choice. Please try again.")
        except ValueError:
            print("Please enter a number.")

def display_visualizations_menu():
    """Display the visualizations menu"""
    print_section("VISUALIZATIONS")
    
    options = [
        "Basic Visualizations (ablation_visualizations/)",
        "Enhanced Visualizations (enhanced_visualizations/)",
        "Specialized Ablation Visualizations (enhanced_ablation_visualizations/)",
        "Back to Main Menu"
    ]
    
    choice = display_menu_and_get_choice(options)
    
    if choice == 1:
        open_visualizations("ablation_visualizations")
    elif choice == 2:
        open_visualizations("enhanced_visualizations")
    elif choice == 3:
        open_visualizations("enhanced_ablation_visualizations")
    elif choice == 4:
        return
    elif choice == 0:
        sys.exit()

def open_visualizations(directory):
    """Open visualizations from a specific directory"""
    clear_screen()
    print_section(f"VISUALIZATIONS IN {directory.upper()}")
    
    try:
        files = [f for f in os.listdir(directory) if f.endswith(('.png', '.jpg'))]
        
        if not files:
            print(f"No visualization files found in {directory}/")
            input("\nPress Enter to continue...")
            return
        
        options = files + ["Back to Visualizations Menu"]
        
        choice = display_menu_and_get_choice(options)
        
        if choice == 0:
            sys.exit()
        elif choice == len(options):
            return
        else:
            filepath = os.path.abspath(os.path.join(directory, files[choice-1]))
            print(f"\nOpening visualization: {files[choice-1]}")
            try:
                webbrowser.open(f"file:///{filepath}")
            except Exception as e:
                print(f"Error opening file: {e}")
            
            time.sleep(1)  # Give browser a moment to open
            open_visualizations(directory)
    except Exception as e:
        print(f"Error: {e}")
        input("\nPress Enter to continue...")

def display_findings_menu():
    """Display the findings menu"""
    print_section("ANALYSIS FINDINGS")
    
    options = [
        "Comprehensive Analysis Report (Markdown)",
        "Comprehensive Analysis Report (HTML)",
        "Ablation Findings Table",
        "Back to Main Menu"
    ]
    
    choice = display_menu_and_get_choice(options)
    
    if choice == 1:
        try:
            filepath = os.path.abspath("analysis_output/comprehensive_analysis.md")
            print(f"\nOpening report: {filepath}")
            webbrowser.open(f"file:///{filepath}")
        except Exception as e:
            print(f"Error opening file: {e}")
    elif choice == 2:
        try:
            filepath = os.path.abspath("analysis_output/comprehensive_analysis.html")
            print(f"\nOpening report: {filepath}")
            webbrowser.open(f"file:///{filepath}")
        except Exception as e:
            print(f"Error opening file: {e}")
    elif choice == 3:
        try:
            filepath = os.path.abspath("ablation_findings_table.md")
            print(f"\nOpening table: {filepath}")
            webbrowser.open(f"file:///{filepath}")
        except Exception as e:
            print(f"Error opening file: {e}")
    elif choice == 4:
        return
    elif choice == 0:
        sys.exit()
    
    time.sleep(1)  # Give browser a moment to open
    display_findings_menu()

def display_run_analysis_menu():
    """Display menu to run analysis scripts"""
    print_section("RUN ANALYSIS SCRIPTS")
    
    options = [
        "Run Basic Ablation Visualizations (visualize_ablation_findings.py)",
        "Run Enhanced Policy Visualizations (enhanced_visualizations.py)",
        "Run Enhanced Ablation Visualizations (enhanced_ablation_visualization.py)",
        "Run Comprehensive Analysis (analyze_ablation_findings.py)",
        "Run All Scripts",
        "Back to Main Menu"
    ]
    
    choice = display_menu_and_get_choice(options)
    
    if choice == 1:
        run_script("visualize_ablation_findings.py")
    elif choice == 2:
        run_script("enhanced_visualizations.py")
    elif choice == 3:
        run_script("enhanced_ablation_visualization.py")
    elif choice == 4:
        run_script("analyze_ablation_findings.py")
    elif choice == 5:
        run_all_scripts()
    elif choice == 6:
        return
    elif choice == 0:
        sys.exit()
    
    input("\nPress Enter to continue...")
    display_run_analysis_menu()

def run_script(script_name):
    """Run a Python script"""
    print(f"\nRunning {script_name}...")
    os.system(f"py {script_name}")

def run_all_scripts():
    """Run all analysis scripts"""
    print("\nRunning all analysis scripts...")
    scripts = [
        "visualize_ablation_findings.py", 
        "enhanced_visualizations.py", 
        "enhanced_ablation_visualization.py", 
        "analyze_ablation_findings.py"
    ]
    
    for script in scripts:
        print(f"\n--- Running {script} ---")
        os.system(f"py {script}")
        print(f"--- Completed {script} ---\n")

def main_menu():
    """Display the main menu"""
    while True:
        clear_screen()
        print_header()
        
        options = [
            "View Visualizations",
            "View Analysis Findings",
            "Run Analysis Scripts"
        ]
        
        choice = display_menu_and_get_choice(options)
        
        if choice == 1:
            display_visualizations_menu()
        elif choice == 2:
            display_findings_menu()
        elif choice == 3:
            display_run_analysis_menu()
        elif choice == 0:
            print("\nExiting. Thank you for using the Bitcoin Policy Analysis tool!")
            sys.exit()

def check_requirements():
    """Check if required files and directories exist"""
    required_dirs = ['ablation_visualizations', 'enhanced_visualizations', 
                   'enhanced_ablation_visualizations', 'analysis_output']
    
    missing = [d for d in required_dirs if not os.path.exists(d)]
    
    if missing:
        print("Some required directories are missing. You may need to run the analysis scripts first.")
        print("Missing directories:", ", ".join(missing))
        
        run_missing = input("\nWould you like to run all analysis scripts now? (y/n): ")
        if run_missing.lower() == 'y':
            run_all_scripts()
        
        print("\nContinuing to main menu...")
        time.sleep(2)

if __name__ == "__main__":
    check_requirements()
    main_menu()

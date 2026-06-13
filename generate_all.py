import os
import subprocess
import sys

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)

    print("--- STEP 1: Creating Jupyter Notebook ---")
    try:
        import create_notebook
        create_notebook.create_notebook()
    except Exception as e:
        print(f"Error creating notebook: {e}")
        sys.exit(1)

    print("\n--- STEP 2: Running Forecasting Pipeline ---")
    try:
        import run_forecasting
        run_forecasting.run_pipeline()
    except Exception as e:
        print(f"Error running forecasting pipeline: {e}")
        sys.exit(1)

    print("\nAll assets successfully generated!")

if __name__ == "__main__":
    main()

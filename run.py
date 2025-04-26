#!/usr/bin/env python3
import os
import subprocess
import sys
import webbrowser
from pathlib import Path

def main():
    """Run the parking detection app."""
    # Get the directory of this script
    script_dir = Path(__file__).parent.absolute()
    
    # Path to Streamlit app
    app_path = script_dir / "src" / "app.py"
    
    # Check if the app file exists
    if not app_path.exists():
        print(f"Error: App file not found at {app_path}")
        return 1
    
    print("Starting Parking Occupancy Detection App...")
    
    # Start Streamlit with the app
    cmd = [sys.executable, "-m", "streamlit", "run", str(app_path), "--server.address=0.0.0.0"]
    
    # Open browser
    webbrowser.open("http://localhost:8501")
    
    # Run the command
    try:
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        print("\nApp stopped.")
    except subprocess.CalledProcessError as e:
        print(f"Error running app: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 
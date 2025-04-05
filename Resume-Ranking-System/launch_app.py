import os
import sys
import subprocess
import time
from threading import Thread
import webbrowser

def run_backend():
    """Run the FastAPI backend server."""
    print("Starting backend server...")
    backend_path = os.path.join("Resume-Ranking-System", "backend", "main.py")
    
    # Start the backend server
    backend_proc = subprocess.Popen(
        [sys.executable, backend_path],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1
    )
    
    # Read and print backend output in a separate thread
    def read_output():
        for line in backend_proc.stdout:
            print(f"[Backend] {line.strip()}")
    
    Thread(target=read_output, daemon=True).start()
    return backend_proc

def run_frontend():
    """Run the frontend server."""
    print("Starting frontend server...")
    frontend_script = os.path.join("Resume-Ranking-System", "frontend", "launch_frontend.py")
    
    # Start the frontend server
    frontend_proc = subprocess.Popen(
        [sys.executable, frontend_script],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1
    )
    
    # Read and print frontend output in a separate thread
    def read_output():
        for line in frontend_proc.stdout:
            print(f"[Frontend] {line.strip()}")
    
    Thread(target=read_output, daemon=True).start()
    return frontend_proc

def main():
    """Main function to run the application."""
    try:
        # Start the backend first
        backend_proc = run_backend()
        
        # Wait for the backend to initialize
        print("Waiting for backend to initialize...")
        time.sleep(5)
        
        # Start the frontend
        frontend_proc = run_frontend()
        
        # Wait a moment for the frontend to start
        time.sleep(2)
        
        # Open browser
        print("Opening browser to http://localhost:5500")
        webbrowser.open("http://localhost:5500")
        
        # Wait for Ctrl+C
        print("\nPress Ctrl+C to stop the servers")
        backend_proc.wait()
        
    except KeyboardInterrupt:
        print("\nShutting down servers...")
        # The processes will be terminated when the script exits
        sys.exit(0)

if __name__ == "__main__":
    main() 
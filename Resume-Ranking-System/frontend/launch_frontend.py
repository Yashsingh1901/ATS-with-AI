import os
import http.server
import socketserver
import webbrowser
from threading import Timer

# Configuration
PORT = 5500
DIRECTORY = os.path.dirname(os.path.abspath(__file__))

class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=DIRECTORY, **kwargs)
    
    def log_message(self, format, *args):
        # Customize logging to be more informative
        print(f"[Frontend Server] - {args[0]} {args[1]} {args[2]}")
    
    def end_headers(self):
        # Add CORS headers
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        super().end_headers()

def open_browser():
    """Open browser after a short delay"""
    webbrowser.open(f'http://localhost:{PORT}')

def main():
    # Change to the directory containing this script
    os.chdir(DIRECTORY)
    
    print(f"Starting frontend server at http://localhost:{PORT}")
    print(f"Serving files from: {DIRECTORY}")
    
    # Schedule browser opening
    Timer(1.5, open_browser).start()
    
    # Start server
    with socketserver.TCPServer(("", PORT), Handler) as httpd:
        print("Server started. Press Ctrl+C to stop.")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nShutting down server...")
            httpd.shutdown()

if __name__ == "__main__":
    main() 
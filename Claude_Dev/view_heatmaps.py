from http.server import HTTPServer, SimpleHTTPRequestHandler
import webbrowser
from pathlib import Path
import sys

# Create a basic HTML template
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Optimization Heatmaps</title>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/react/18.2.0/umd/react.production.min.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/react-dom/18.2.0/umd/react-dom.production.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <link href="https://cdn.jsdelivr.net/npm/tailwindcss@2.2.19/dist/tailwind.min.css" rel="stylesheet">
</head>
<body>
    <div id="root"></div>
    <script src="optimization-heatmaps.js"></script>
</body>
</html>
"""

class Handler(SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/':
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            self.wfile.write(HTML_TEMPLATE.encode())
        else:
            super().do_GET()

def main():
    # Check if optimization results exist
    results_path = Path("Optimization Results/detailed_results.csv")
    if not results_path.exists():
        print("Error: Could not find optimization results.")
        print("Please run the optimization first to generate results.")
        sys.exit(1)

    # Start server on a random port
    server = HTTPServer(('localhost', 0), Handler)
    port = server.server_port

    # Open browser
    url = f'http://localhost:{port}'
    print(f"Opening visualization in browser at {url}")
    webbrowser.open(url)

    # Run server
    try:
        print("Press Ctrl+C to stop the server")
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down server...")
        server.server_close()

if __name__ == "__main__":
    main()
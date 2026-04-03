import httpx
import sys

base_url = "http://127.0.0.1:8000"

routes = [
    "/prototype-summary",
    "/api/prototype-stats",
    "/api/quality-report"
]

def verify_routes():
    print(f"Verifying routes on {base_url}...")
    for route in routes:
        try:
            # We assume the server is running. If not, we might need to start it.
            # But the user is likely running it.
            res = httpx.get(base_url + route)
            if res.status_code == 200:
                print(f"[OK] {route}")
            else:
                print(f"[FAIL] {route} (Status: {res.status_code})")
        except Exception as e:
            print(f"[ERROR] {route}: {e}")

if __name__ == "__main__":
    verify_routes()

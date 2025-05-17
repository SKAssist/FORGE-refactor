from datetime import datetime

def log_results(records):
    print(f"--- LOG [{datetime.now()}] ---")
    for u, d in records:
        print(f"{u} used the app for {d} minutes.")

inputs = [
    {'user': 'Ivan', 'duration': 15},
    {'user': 'Jade', 'duration': 20},
    {'duration': 10}
]
res = analyze_sessions(inputs)
    log_results(res)
else:
    print("No valid sessions found.")
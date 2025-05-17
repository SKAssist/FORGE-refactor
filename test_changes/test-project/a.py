raw_sessions = [
    {'user': 'Alice', 'duration': 30},
    {'user': 'Bob', 'duration': 0},
    {'user': 'Charlie', 'duration': 45},
    None,
    {'user': 'Dana'}  # missing duration
]
summary = analyze_sessions(raw_sessions, log=True)
print("Summary Length:", len(summary))
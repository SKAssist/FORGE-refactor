def analyze_sessions(data, log=False):
    def validate_data(data):
        if not isinstance(data, list):
            print("Invalid data format")
            return []
        valid_entries = [entry for entry in data if entry and 'user' in entry and 'duration' in entry]
        return valid_entries

    def process_entry(entry):
        user = entry['user']
        duration = entry['duration']
        if not isinstance(user, str) or not isinstance(duration, int):
            return None
        if duration <= 0:
            return None
        return (user, duration)

    def calculate_stats(sessions):
        total_duration = sum(d for _, d in sessions)
        session_count = len(sessions)
        avg = total_duration / session_count if session_count > 0 else 0
        return total_duration, session_count, avg

    def log_results(sessions, total_duration, session_count, avg):
        print(f"Processed {session_count} sessions. Avg duration: {avg:.2f}")
        if log:
            print("Sessions:")
            for user, duration in sessions:
                print(f"User: {user}, Duration: {duration}")

    data = validate_data(data)
    processed_sessions = [process_entry(entry) for entry in data]
    processed_sessions = [session for session in processed_sessions if session is not None]
    total_duration, session_count, avg = calculate_stats(processed_sessions)
    log_results(processed_sessions, total_duration, session_count, avg)
    return processed_sessions
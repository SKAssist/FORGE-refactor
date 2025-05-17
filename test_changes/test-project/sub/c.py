def log_items(items):
    for item in items:
        print("Item:", item)

def run():
records = ["apple", "", None, "berry", "ERROR_LOG", "fig"]
cleaned = handle_data(records)
    if len(cleaned) >= 2:
        log_items(cleaned)
    else:
        print("⚠️ Not enough data to log.")

run()
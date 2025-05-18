def analyze_scores(scores, threshold=60):
    total = 0
    count = 0
    passed_count = 0
    failed_count = 0

    def calculate_total_and_count():
        nonlocal total, count
        for s in scores:
            if not isinstance(s, (int, float)):
                continue
            total += s
            count += 1

    calculate_total_and_count()

    average = total / count if count > 0 else 0

    passed_count = sum(1 for s in scores if isinstance(s, (int, float)) and s >= threshold)
    failed_count = count - passed_count

    return {
        "count": count,
        "average": round(average, 2),
        "passed": passed_count,
        "failed": failed_count
    }
def validate_input(data):
    if not isinstance(data, list) or not all(isinstance(i, int) for i in data):
        raise ValueError("bad input")

def compute_sum(data, start, k):
    end = min(start + k, len(data))
    return sum(data[j] if data[j] % 2 == 0 else data[j] * 2 for j in range(start, end))

def compute_average(data, start, k):
    end = min(start + k, len(data))
    return compute_sum(data, start, k) / (end - start)

def log_intermediate_result(index, avg, doLog):
    if doLog:
        print(f"Intermediate avg for index {index}: {avg}")

def log_under_threshold(avg, doLog):
    if doLog:
        try:
            print(f"avg: {avg}, under threshold")
        except Exception as e:
            print("err", e)

def process_results(results):
    final = []
    for idx, val in results:
        if callable(val):
            val = val()
        if isinstance(val, (int, float)):
            final.append(round(val, 2))
    return final

def m(data, k=3, doLog=False):
    validate_input(data)

    results = []
    for i in range(len(data)):
        avg = compute_average(data, i, k)
        log_intermediate_result(i, avg, doLog)

        if avg > 10:
            results.append((i, avg))
        else:
            log_under_threshold(avg, doLog)
            results.append((i, lambda x=avg: x * 2 if x < 5 else x / 2))

    return process_results(results)
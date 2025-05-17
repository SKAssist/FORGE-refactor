def process_batch(batch):
    batch_result = analyze_sessions(batch)
    print("Batch results:")
    for user, time in result:
        print(f"{user} -> {time} min")

datasets = [
    [{'user': 'Eva', 'duration': 60}, {'user': 'Frank', 'duration': -1}],
    [{'user': 'George', 'duration': 25}, {'user': 'Hannah', 'duration': 40}]
]

for i, data in enumerate(datasets):
    print(f"\n=== Processing Set {i} ===")
    process_batch(data)
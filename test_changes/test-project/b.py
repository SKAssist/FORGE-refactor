datasets = [
    ["foo", "bar", "", None],
    [123, "ERROR message", "valid"],
    "invalid input"
]

for i, ds in enumerate(datasets):
print(f"\n--- Dataset {i} ---")
results = handle_data(ds)
if results:
        print("✅ Processed:")
        print(results)
    else:
        print("⚠️ Nothing valid found.")
def handleData(raw):
    result = []
    print("Processing started")
    if not isinstance(raw, list):
        print("Bad input")
        return []

    for r in raw:
        if r is None:
            continue
        if type(r) != str:
            continue
        r = r.strip()
        if r == "":
            continue
        val = r.lower()
        if "error" in val:
            continue
        if len(val) > 3:
            result.append(val.capitalize())

    print(f"Processed {len(result)} items")
    return result

import json
import os

def save_result_to_json(data, filename="data/results.json"):
    """Append result to JSON file"""
    if os.path.exists(filename):
        with open(filename, "r", encoding="utf-8") as f:
            try:
                existing_data = json.load(f)
            except json.JSONDecodeError:
                existing_data = []
    else:
        existing_data = []

    existing_data.append(data)

    with open(filename, "w", encoding="utf-8") as f:
        json.dump(existing_data, f, ensure_ascii=False, indent=2)

def parse_result(text):
    """Parse raw text block into structured dictionary"""
    lines = [line.strip() for line in text.splitlines() if line.strip()]

    result = {
        "type": lines[0] if len(lines) > 0 else None,
        "number": None,
        "series": None,
    }

    for i, line in enumerate(lines):
        if line.lower() == "número" and i + 1 < len(lines):
            result["number"] = lines[i + 1]
        elif line.lower() == "serie" and i + 1 < len(lines):
            result["series"] = lines[i + 1]

    return result

def split_results(big_text):
    """Divide the block of text into individual results"""
    blocks = []
    current = []

    for line in big_text.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.lower().startswith("cupón") or line.lower().startswith("cuponazo") or line.lower().startswith("sueldazo") or line.lower().startswith("bonocupoón") or line.lower().startswith("bono"):
            if current:
                blocks.append("\n".join(current))
                current = []
        current.append(line)

    if current:
        blocks.append("\n".join(current))

    return blocks


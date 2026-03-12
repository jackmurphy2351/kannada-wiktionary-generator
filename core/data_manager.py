import os
import json

JSON_FILE = 'data/verified_kannada_entries.json'


def load_ground_truth():
    if os.path.exists(JSON_FILE) and os.path.getsize(JSON_FILE) > 0:
        with open(JSON_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}


def save_to_ground_truth(word, entry):
    data = load_ground_truth()
    data[word] = entry

    # Ensure the directory exists just in case
    os.makedirs(os.path.dirname(JSON_FILE), exist_ok=True)

    with open(JSON_FILE, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


def format_time(seconds):
    seconds = int(seconds)
    if seconds < 60: return f"{seconds}s"
    minutes = seconds // 60
    remaining_seconds = seconds % 60
    if minutes < 60: return f"{minutes}m {remaining_seconds}s"
    return f"{minutes // 60}h {minutes % 60}m {remaining_seconds}s"
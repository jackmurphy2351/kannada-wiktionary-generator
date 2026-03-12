import yaml
import os
import time
import json

from core.wiktionary_api import check_wiktionary_entry_exists
from core.scraper import source_etymology
from core.linguistics import get_dynamic_morphology_block

STAGING_FILE = 'data/staged_entries.json'
DISCARDED_FILE = 'data/discarded_entries.json'
BATCH_SIZE = 50


def load_alar_data():
    """Loads the Alar dictionary from the YAML file."""
    with open('data/alar.yml', 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def clean_definition(def_text):
    """A basic cleaner to handle Alar's verbose strings."""
    if ':' in def_text:
        def_text = def_text.split(':')[-1]
    elif ';' in def_text:
        def_text = def_text.split(';')[0]
    return def_text.strip().strip('.')


def group_alar_data(raw_alar_data):
    """Groups multiple YAML entries and definitions under a single word."""
    grouped = {}
    for item in raw_alar_data:
        word = item.get("entry", "").strip()
        defs_list = item.get("defs", [])

        if not word or not defs_list:
            continue

        if word not in grouped:
            grouped[word] = {}

        for def_obj in defs_list:
            raw_def = def_obj.get("entry", "UNKNOWN")
            cleaned_def = clean_definition(raw_def)
            pos = def_obj.get("type", "noun").title()

            # Initialize the POS list if it doesn't exist
            if pos not in grouped[word]:
                grouped[word][pos] = []

            # Avoid duplicate definitions for the same POS
            if cleaned_def not in grouped[word][pos]:
                grouped[word][pos].append(cleaned_def)

    return grouped


def build_mvp_wikitext(word, grouped_defs, etymology_line, cognate_line):
    """Constructs the base Wikitext string with support for multiple POS."""
    formatted_etymology = f"Borrowed from {etymology_line}." if "{{bor" in etymology_line else f"From {etymology_line}." if "{{inh" in etymology_line else etymology_line
    if cognate_line:
        formatted_etymology += f" {cognate_line}"

    wikitext = f"==Kannada==\n\n===Etymology===\n{formatted_etymology}\n\n===Pronunciation===\n* {{{{kn-IPA|{word}}}}}\n"

    # Iterate through each Part of Speech for the word
    for pos, definitions in grouped_defs.items():
        if pos.lower() == "adjective":
            pos_template = "adj"
        elif pos.lower() == "adverb":
            pos_template = "adv"
        else:
            pos_template = pos.lower()

        # Generate morphology strictly for this specific POS
        morphology_block = get_dynamic_morphology_block(word, [pos])
        if "IRREGULAR_CHECK" in morphology_block:
            stem = word.removesuffix("ು")
            morphology_block = morphology_block.replace("IRREGULAR_CHECK", f"{{{{kn-conj-u|{word}|{stem}ಿ|{stem}ಿದ}}}}")

        formatted_morphology = f"\n\n{morphology_block}" if morphology_block.strip() else ""

        # Build the numbered definition list
        defs_wikitext = "\n".join([f"# [[{d}]]" for d in definitions])

        wikitext += f"\n==={pos}===\n{{{{kn-{pos_template}}}}}\n{defs_wikitext}{formatted_morphology}\n"

    wikitext += "\n===References===\n* {{R:kn:Alar}}\n* {{R:kn:Kittel}}"
    return wikitext


def run_batch_processor():
    raw_alar_data = load_alar_data()

    print("Grouping Alar dataset...")
    grouped_alar_data = group_alar_data(raw_alar_data)
    print(f"Grouped into {len(grouped_alar_data)} unique words.")

    if os.path.exists(STAGING_FILE):
        with open(STAGING_FILE, 'r', encoding='utf-8') as f:
            staged_data = json.load(f)
    else:
        staged_data = {}

    if os.path.exists(DISCARDED_FILE):
        with open(DISCARDED_FILE, 'r', encoding='utf-8') as f:
            discarded_data = json.load(f)
    else:
        discarded_data = {}

    words_processed = 0

    for word, grouped_defs in grouped_alar_data.items():
        if words_processed >= BATCH_SIZE:
            print(f"Batch limit of {BATCH_SIZE} reached. Ready for review.")
            break

        if word in staged_data or word in discarded_data:
            continue

        if check_wiktionary_entry_exists(word):
            print(f"Skipping '{word}': Already exists on Wiktionary.")
            continue

        print(f"Processing '{word}'...")

        etymology_line, cognate_text, _ = source_etymology(word)

        # Build the dynamic Wikitext
        mvp_wikitext = build_mvp_wikitext(word, grouped_defs, etymology_line, cognate_text)

        # Grab the very first POS and definition to serve as the "Primary" for the UI and Ollama prompt
        primary_pos = list(grouped_defs.keys())[0]
        primary_definition = grouped_defs[primary_pos][0]

        staged_data[word] = {
            "definition": primary_definition,  # Keeps bulk_review_ui.py happy!
            "pos": primary_pos,
            "wikitext": mvp_wikitext,
            "status": "pending_review"
        }

        words_processed += 1
        time.sleep(1)

    os.makedirs('data', exist_ok=True)
    with open(STAGING_FILE, 'w', encoding='utf-8') as f:
        json.dump(staged_data, f, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    run_batch_processor()
import yaml
import os
import time
import json

# Import the necessary functions from your core modules
from core.wiktionary_api import check_wiktionary_entry_exists
from core.scraper import source_etymology
from core.linguistics import get_dynamic_morphology_block

STAGING_FILE = 'data/staged_entries.json'
DISCARDED_FILE = 'data/discarded_entries.json' # 'Trash can' for discarded entries
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


def build_mvp_wikitext(word, pos, definition, etymology_line, cognate_line, morphology_block):
    """Constructs the base Wikitext string."""
    formatted_etymology = f"Borrowed from {etymology_line}." if "{{bor" in etymology_line else f"From {etymology_line}." if "{{inh" in etymology_line else etymology_line
    if cognate_line:
        formatted_etymology += f" {cognate_line}"

    # Fix 1: Map "Adjective" and "Adverb" to their abbreviated Wiktionary templates
    if pos.lower() == "adjective":
        pos_template = "adj"
    elif pos.lower() == "adverb":
        pos_template = "adv"
    else:
        pos_template = pos.lower()

    # Fix 2: Conditionally format the morphology block to prevent multi-line gaps
    formatted_morphology = f"\n\n{morphology_block}" if morphology_block.strip() else ""

    return f"""==Kannada==

===Etymology===
{formatted_etymology}

===Pronunciation===
* {{{{kn-IPA|{word}}}}}

==={pos}===
{{{{kn-{pos_template}}}}}
# [[{definition}]]{formatted_morphology}

===References===
* {{{{R:kn:Alar}}}}
* {{{{R:kn:Kittel}}}}"""


def run_batch_processor():
    alar_data = load_alar_data()

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

    for item in alar_data:
        if words_processed >= BATCH_SIZE:
            print(f"Batch limit of {BATCH_SIZE} reached. Ready for review.")
            break

        word = item.get("entry", "").strip()
        defs_list = item.get("defs", [])

        if not word or not defs_list:
            continue

        if word in staged_data or word in discarded_data:
            continue

        if check_wiktionary_entry_exists(word):
            print(f"Skipping '{word}': Already exists on Wiktionary.")
            continue

        print(f"Processing '{word}'...")

        primary_def_obj = defs_list[0]
        raw_definition = primary_def_obj.get("entry", "UNKNOWN")
        definition = clean_definition(raw_definition)

        pos = primary_def_obj.get("type", "noun").title()

        etymology_line, cognate_text, _ = source_etymology(word)
        morphology_block = get_dynamic_morphology_block(word, [pos])

        if "IRREGULAR_CHECK" in morphology_block:
            stem = word.removesuffix("ು")
            morphology_block = morphology_block.replace("IRREGULAR_CHECK", f"{{{{kn-conj-u|{word}|{stem}ಿ|{stem}ಿದ}}}}")

        mvp_wikitext = build_mvp_wikitext(word, pos, definition, etymology_line, cognate_text, morphology_block)

        staged_data[word] = {
            "definition": definition,
            "pos": pos,
            "wikitext": mvp_wikitext,
            "status": "pending_review"
        }

        words_processed += 1
        time.sleep(1)

    # Ensure the data directory exists before saving
    os.makedirs('data', exist_ok=True)
    with open(STAGING_FILE, 'w', encoding='utf-8') as f:
        json.dump(staged_data, f, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    run_batch_processor()
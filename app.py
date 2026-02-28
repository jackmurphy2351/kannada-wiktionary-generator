import streamlit as st
import json
import ollama
import os
import time
from dotenv import load_dotenv

load_dotenv()

# --- DATA MANAGEMENT ---
JSON_FILE = 'verified_kannada_entries.json'


def load_ground_truth():
    if os.path.exists(JSON_FILE) and os.path.getsize(JSON_FILE) > 0:
        with open(JSON_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}


def save_to_ground_truth(word, entry):
    data = load_ground_truth()
    data[word] = entry
    with open(JSON_FILE, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


def format_time(seconds):
    seconds = int(seconds)
    if seconds < 60: return f"{seconds}s"
    minutes = seconds // 60
    remaining_seconds = seconds % 60
    if minutes < 60: return f"{minutes}m {remaining_seconds}s"
    return f"{minutes // 60}h {minutes % 60}m {remaining_seconds}s"


def get_dynamic_morphology_block(word, pos_categories):
    blocks = []
    last_char = word[-1] if word else ""
    for pos in pos_categories:
        if pos == "Noun":
            header = "====Declension===="
            if last_char == "ು":
                stem = word.removesuffix("ು")
                template = f"{{{{kn-decl-u|{word}|{stem}}}}}"
            elif last_char in {"ಿ", "ೆ", "ೈ"}:
                template = f"{{{{kn-decl-e-i-ai|{word}|{word}}}}}"
            else:
                template = f"{{{{kn-decl-a|{word}}}}}"
            blocks.append(f"{header}\n{template}")
        elif pos == "Verb":
            header = "====Conjugation===="
            if word.endswith("ಕೊಳ್ಳು"):
                prefix = word.removesuffix("ಕೊಳ್ಳು")
                template = f"{{{{kn-conj-koḷḷu|{prefix}}}}}"
            elif word.endswith("ಿಸು"):
                stem = word.removesuffix("ು")
                template = f"{{{{kn-conj-isu|{word}|{stem}}}}}"
            elif last_char in {"ಿ", "ೆ", "ೈ"}:
                template = f"{{{{kn-conj-e-i-other|{word}|{word}ಯ|{word}ದ|{word}}}}}"
            else:
                template = "IRREGULAR_CHECK"
            blocks.append(f"{header}\n{template}")
    return "\n\n".join(blocks) if blocks else ""


def get_few_shot_examples(current_ground_truth, pos_categories, target_word, count=2):
    if not current_ground_truth: return ""
    examples = []
    for word, wikitext in current_ground_truth.items():
        if len(examples) >= count: break
        if any(f"==={pos}===" in wikitext for pos in pos_categories):
            examples.append(f"Word: {word}\nOutput:\n{wikitext}\n---")
    return "\n".join([f"\nExample {i + 1}:\n{ex}" for i, ex in enumerate(examples)])


# --- SYSTEM PROMPTS ---

DRAFTER_PROMPT = """
You are an expert Lexicographer. TASK: Fill placeholders in the provided Wiktionary skeleton.

1. ETYMOLOGY: If the skeleton contains the placeholder `DEVANAGARI_WORD`, replace it with the target word written in Devanagari script (e.g., replace `DEVANAGARI_WORD` with `स्नानगृह`). Do not change it if it says `{{rfe|kn}}`.
2. USAGE_NOTES_ENTRY: Replace with exactly ONE formal example: `* {{ux|kn|KANNADA_SCRIPT_HERE|tr=ISO_15919_ROMAN|t=ENGLISH_TRANSLATION}}`. 
   - CRITICAL: 'tr=' MUST use Roman script. 't=' MUST be in English.

CRITICAL STRUCTURAL RULES:
- FULL OUTPUT: You MUST output the ENTIRE Wiktionary entry from top to bottom. Do NOT just output the filled placeholders.
- HEADERS: Do NOT delete the headers (e.g., ==Kannada==, ===Etymology===, ===Noun===).
- DEFINITIONS: Do NOT change the English words in the `# [[ ]]` line. Leave the English meaning exactly as it is in the skeleton.
"""

LOGIC_AUDITOR_PROMPT = """
You are a strict Linguistic QA Editor. Fix errors and output the ENTIRE raw Wikitext.

1. DEVANAGARI AUDIT: If the etymology line contains the literal text 'DEVANAGARI_WORD', you MUST replace that text with the target Kannada word translated into actual Devanagari script.
2. PARAMETER AUDIT: Ensure the first parameter of `{{ux}}` contains Kannada script. Ensure 'tr=' is Roman and 't=' is English. If they contain Kannada, translate/transliterate them correctly.
3. DEFINITION AUDIT: Ensure the '#' line contains the provided English meaning in brackets. Change it back to English if the draft used Kannada.
4. STRUCTURE (CRITICAL): You MUST ensure the output starts exactly with `==Kannada==` and retains ALL headers (e.g., ===Etymology===, ====Usage notes====). Do not delete them.
"""

TRANSLIT_AUDITOR_PROMPT = """
TASK: Fix the Roman transliteration (tr=) in the provided Wikitext. Output the ENTIRE raw Wikitext.

1. CHARACTER-LITERAL: Map character-by-character (e.g., ಳ is ḷ, ಷ is ṣ, ಣ is ṇ). NEVER drop or skip syllables at the end of long words.
2. SANDHI & SUFFIX PRESERVATION: NEVER insert spaces where the Kannada script has none. If the script has a long suffix (e.g., ಸ್ವಾತಂತ್ರ್ಯಕ್ಕಾಗಿ), the transliteration MUST spell out every single letter (e.g., svātantryakkāgi). Do not truncate it or collapse it into sloppy spoken forms (e.g., do NOT write 'svātantryakki').
3. LATIN SCRIPT ONLY: If 'tr=' contains Kannada script, replace it with correct ISO 15919 Roman script.
4. MACRONS: Use ā, ē, ī, ō, ū correctly (e.g., ಅನೇಕ is anēka).
5. CASE: Do not arbitrarily capitalize words in the middle of sentences.

CRITICAL: Output must start exactly with ==Kannada==. Do not wrap in markdown.
"""

# --- APP UI ---
st.set_page_config(page_title="Kannada Wiktionary Gen", page_icon="🌿")

DRAFTER_MODEL = 'translategemma:27b'
LOGIC_MODEL = 'gemma2:9b'
PROOFREADER_MODEL = 'translategemma:4b'

st.title("Kannada Wiktionary Generator")
word = st.text_input("Enter word:")
translation = st.text_input("Enter translation:")

pos_categories = st.multiselect("POS:", ["Noun", "Verb", "Adjective", "Adverb"], default=["Noun"])

if word:
    ground_truth = load_ground_truth()
    if word in ground_truth:
        st.success("Found!")
        st.text_area("Entry:", ground_truth[word], height=400)
    else:
        if st.button("Generate Wikitext"):
            progress_bar = st.progress(0)
            status_text = st.empty()
            timer_text = st.empty()
            try:
                start_time = time.time()
                morphology_block = get_dynamic_morphology_block(word, pos_categories)
                if "IRREGULAR_CHECK" in morphology_block:
                    stem = word.removesuffix("ು")
                    morphology_block = morphology_block.replace("IRREGULAR_CHECK",
                                                                f"{{{{kn-conj-u|{word}|{stem}ಿ|{stem}ಿದ}}}}")

                primary_pos = pos_categories[0] if pos_categories else "Noun"
                pos_template = f"{{{{kn-{primary_pos.lower()}}}}}"
                formatted_translation = ", ".join([f"[[{t.strip()}]]" for t in translation.split(",") if t.strip()])

                # --- PYTHON ETYMOLOGY LOGIC ---
                sanskrit_triggers = set("ಖಘಛಝಠಢಥಧಫಭಋೃಶಷಃಙಞ")
                if any(char in word for char in sanskrit_triggers):
                    etymology_line = "{{bor|kn|sa|DEVANAGARI_WORD}}"
                else:
                    etymology_line = "{{rfe|kn}}"

                wiktionary_skeleton = f"""==Kannada==

===Etymology===
{etymology_line}

===Pronunciation===
* {{{{kn-IPA|{word}}}}}

==={primary_pos}===
{pos_template}
# {formatted_translation}

====Usage notes====
USAGE_NOTES_ENTRY

{morphology_block}

===References===
* {{{{R:kn:Alar}}}}"""

                examples_block = get_few_shot_examples(ground_truth, pos_categories, word)

                # STEP 1: DRAFTING
                status_text.text("Step 1/3: Drafting...")
                progress_bar.progress(10)
                drafter_user_content = f"### Context Examples:\n{examples_block}\n\n### Target Skeleton to Fill:\n{wiktionary_skeleton}"
                draft = ""
                for chunk in ollama.chat(model=DRAFTER_MODEL, messages=[{'role': 'system', 'content': DRAFTER_PROMPT},
                                                                        {'role': 'user',
                                                                         'content': drafter_user_content}],
                                         stream=True):
                    draft += chunk['message']['content']
                    timer_text.markdown(f"**⏱️ Total Time:** `{format_time(time.time() - start_time)}`")

                # STEP 2: LOGIC AUDIT
                status_text.text("Step 2/3: Logic Audit...")
                progress_bar.progress(40)
                logic_entry = ""

                # CRITICAL PIPELINE FIX: Passing the actual context to the Auditor!
                logic_user_content = (
                    f"Target Word: {word}\n"
                    f"Required English Meaning: {translation}\n\n"
                    f"Draft Wikitext to Correct:\n{draft}"
                )

                for chunk in ollama.chat(model=LOGIC_MODEL,
                                         messages=[{'role': 'system', 'content': LOGIC_AUDITOR_PROMPT},
                                                   {'role': 'user', 'content': logic_user_content}], stream=True):
                    logic_entry += chunk['message']['content']
                    timer_text.markdown(f"**⏱️ Total Time:** `{format_time(time.time() - start_time)}`")

                # STEP 3: TRANSLITERATION AUDIT
                status_text.text("Step 3/3: Transliteration Audit...")
                progress_bar.progress(70)
                final_entry = ""
                for chunk in ollama.chat(model=PROOFREADER_MODEL,
                                         messages=[{'role': 'system', 'content': TRANSLIT_AUDITOR_PROMPT},
                                                   {'role': 'user', 'content': logic_entry}], stream=True):
                    final_entry += chunk['message']['content']
                    timer_text.markdown(f"**⏱️ Total Time:** `{format_time(time.time() - start_time)}`")

                final_entry = final_entry.replace("```wikitext", "").replace("```", "").strip()
                if "==Kannada==" in final_entry:
                    final_entry = "==Kannada==" + final_entry.split("==Kannada==")[-1]
                st.session_state['current_result'] = final_entry
                progress_bar.progress(100)
                status_text.text("Done!")

            except Exception as e:
                st.error(f"Error: {e}")
            finally:
                time.sleep(1); progress_bar.empty(); status_text.empty(); timer_text.empty()

    if 'current_result' in st.session_state:
        edited_entry = st.text_area("Verify:", st.session_state['current_result'], height=400)
        if st.button("Save"):
            save_to_ground_truth(word, edited_entry)
            st.success("Saved!")
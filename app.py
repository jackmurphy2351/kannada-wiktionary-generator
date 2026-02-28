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
    """
    Returns both the correct header and the template string.
    This completely removes the burden of matching headers to templates from the LLM.
    """
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

# STEP 1: DRAFTING
DRAFTER_PROMPT = """
You are an expert Lexicographer. 
TASK: You will be provided with a partially filled Wiktionary skeleton. You must ONLY fill in the remaining placeholder blocks. Output raw Wikitext ONLY.

INSTRUCTIONS FOR FILLING THE SKELETON:
1. ETYMOLOGY_ENTRY: 
   - IF the target word contains aspirated consonants (ಖ, ಘ, ಛ, ಝ, ಠ, ಢ, ಥ, ಧ, ಫ, ಭ), it is a Sanskrit borrowing. Replace the placeholder with: `{{bor|kn|sa|DEVANAGARI_WORD}}` (e.g., {{bor|kn|sa|भूमि}}).
   - ELSE, it is likely native or unknown. Replace the placeholder with: `{{rfe|kn}}`.
2. USAGE_NOTES_ENTRY: 
   - Replace the placeholder with exactly ONE sophisticated, formal example sentence using the template: `* {{ux|kn|KANNADA_SENTENCE|tr=TRANSLITERATION|t=ENGLISH_TRANSLATION}}`

CRITICAL: Do not alter the headers, the English definitions, or the morphology templates that have already been populated in the skeleton. Do not wrap output in markdown.
"""

# STEP 2: LOGIC & ETYMOLOGY AUDIT
LOGIC_AUDITOR_PROMPT = """
You are a strict Linguistic QA Editor. 
TASK: Fix logic errors in the draft and output ONLY raw Wikitext.

1. DEVANAGARI AUDIT: Check the `{{bor|kn|sa|...}}` template. If the word inside is written in Kannada script (like ಭೂಮಿ), you MUST translate it into Devanagari script (like भूमि).
2. SCRIPT CONTAMINATION: Check the `{{kn-IPA|...}}` template. It MUST contain the target word in Kannada script. Change Malayalam (e.g., ഭൂമി) or Telugu scripts back to Kannada.
3. STRUCTURAL PRESERVATION: Do NOT delete the `====Declension====`, `====Conjugation====`, or `===References===` headers.
"""

# STEP 3: ATOMIC TRANSLITERATION AUDIT
TRANSLIT_AUDITOR_PROMPT = """
TASK: Fix the Roman transliteration (`tr=`) in the provided Wikitext. Output ONLY raw Wikitext. DO NOT change anything else.

TRANSLITERATION RULES (ISO 15919):
1. EXACT CHARACTER MAPPING: Map character-by-character. Respect joined words (sandhi) exactly as written in the Kannada text.
2. DENTALS (NO UNDERDOTS): ತ = t, ದ = d, ನ = n.
3. RETROFLEXES (UNDERDOTS REQUIRED): ಟ = ṭ, ಡ = ḍ, ಣ = ṇ, ಳ = ḷ, ಷ = ṣ.
4. MACRONS: ಆ = ā, ಈ = ī, ಊ = ū, ಏ = ē, ಓ = ō.
5. CASE PRESERVATION: Do not arbitrarily capitalize words in the middle of the transliterated sentence.

CRITICAL: Do not wrap your output in ```wikitext blocks. Output must start exactly with ==Kannada==.
"""

# --- APP UI ---
st.set_page_config(page_title="Kannada Wiktionary Gen", page_icon="🌿")

DRAFTER_MODEL = 'translategemma:27b'
LOGIC_MODEL = 'gemma2:9b'
PROOFREADER_MODEL = 'translategemma:4b'

st.sidebar.title("Model Desk")
st.sidebar.info(f"1. Drafter: `{DRAFTER_MODEL}`\n2. Linguist: `{LOGIC_MODEL}`\n3. Proofreader: `{PROOFREADER_MODEL}`")

st.title("Kannada Wiktionary Generator")
word = st.text_input("Enter word:")
translation = st.text_input("Enter translation:")

if "last_word" not in st.session_state or st.session_state["last_word"] != word:
    st.session_state["last_word"] = word
    st.session_state.pop('current_result', None)

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

                # --- PYTHON DYNAMIC SKELETON GENERATION ---
                morphology_block = get_dynamic_morphology_block(word, pos_categories)
                if "IRREGULAR_CHECK" in morphology_block:
                    stem = word.removesuffix("ು")
                    morphology_block = morphology_block.replace("IRREGULAR_CHECK",
                                                                f"{{{{kn-conj-u|{word}|{stem}ಿ|{stem}ಿದ}}}}")

                # -> MOVED THESE LINES OUTSIDE THE IF BLOCK <-
                # Determine primary POS for the header
                primary_pos = pos_categories[0] if pos_categories else "Noun"
                pos_template = f"{{{{kn-{primary_pos.lower()}}}}}"

                # Handle multiple comma-separated meanings for Wikilinks
                formatted_translation = ", ".join([f"[[{t.strip()}]]" for t in translation.split(",") if t.strip()])

                # Construct the raw skeleton
                wiktionary_skeleton = f"""==Kannada==

===Etymology===
ETYMOLOGY_ENTRY

===Pronunciation===
* {{{{kn-IPA|{word}}}}}

==={primary_pos}===
{pos_template}
# {formatted_translation}

====Usage notes====
USAGE_NOTES_ENTRY

{morphology_block}

===References===
* {{{{R:kn:Alar}}}}
"""
                examples_block = get_few_shot_examples(ground_truth, pos_categories, word)

                # STEP 1: DRAFTING
                status_text.text("Step 1/3: Drafting...")
                progress_bar.progress(10)
                draft = ""

                drafter_user_content = (
                    f"### Context Examples:\n{examples_block}\n\n"
                    f"### Target Skeleton to Fill Out:\n{wiktionary_skeleton}\n"
                )

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

                for chunk in ollama.chat(model=LOGIC_MODEL,
                                         messages=[{'role': 'system', 'content': LOGIC_AUDITOR_PROMPT},
                                                   {'role': 'user', 'content': draft}], stream=True):
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

                # --- PYTHON MARKDOWN BLEED CLEANUP ---
                # Ruthlessly strip any markdown block characters that the models might have added
                final_entry = final_entry.replace("```wikitext", "").replace("```wiktionary", "").replace("```",
                                                                                                          "").strip()

                if "==Kannada==" in final_entry:
                    final_entry = "==Kannada==" + final_entry.split("==Kannada==")[-1]

                st.session_state['current_result'] = final_entry.strip().removesuffix("---").strip()
                progress_bar.progress(100)
                status_text.text("Done!")

            except Exception as e:
                st.error(f"Error: {e}")
            finally:
                time.sleep(1)
                progress_bar.empty()
                status_text.empty()
                timer_text.empty()

    if 'current_result' in st.session_state:
        edited_entry = st.text_area("Verify:", st.session_state['current_result'], height=400)
        if st.button("Save"):
            save_to_ground_truth(word, edited_entry)
            st.success("Saved!")
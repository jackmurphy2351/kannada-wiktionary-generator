import streamlit as st
import json
import ollama
import os
import time
import re
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


def get_few_shot_sentences(current_ground_truth, target_word, count=2):
    if not current_ground_truth: return ""
    examples = []
    for word, wikitext in current_ground_truth.items():
        if len(examples) >= count: break
        match = re.search(r'\{\{ux\|kn\|(.*?)\|tr=.*?\|t=(.*?)\}\}', wikitext)
        if match:
            kn_sent = match.group(1).strip()
            en_sent = match.group(2).strip()
            examples.append(f"Target Word: {word}\nKANNADA: {kn_sent}\nENGLISH: {en_sent}\n---")
    return "\n".join(examples)


def parse_kannada_english(text):
    kn, en = "", ""
    if "KANNADA:" in text and "ENGLISH:" in text:
        kn = text.split("KANNADA:")[1].split("ENGLISH:")[0].replace("```", "").strip()
        en = text.split("ENGLISH:")[1].replace("```", "").strip()
    return kn, en


# --- PYTHON TRANSLITERATION LOGIC ---
def transliterate_kannada_to_iso(text):
    """Deterministically maps Kannada text to ISO 15919 transliteration."""
    KANNADA_CONSONANTS = {
        'ಕ': 'k', 'ಖ': 'kh', 'ಗ': 'g', 'ಘ': 'gh', 'ಙ': 'ṅ',
        'ಚ': 'c', 'ಛ': 'ch', 'ಜ': 'j', 'ಝ': 'jh', 'ಞ': 'ñ',
        'ಟ': 'ṭ', 'ಠ': 'ṭh', 'ಡ': 'ḍ', 'ಢ': 'ḍh', 'ಣ': 'ṇ',
        'ತ': 't', 'ಥ': 'th', 'ದ': 'd', 'ಧ': 'dh', 'ನ': 'n',
        'ಪ': 'p', 'ಫ': 'ph', 'ಬ': 'b', 'ಭ': 'bh', 'ಮ': 'm',
        'ಯ': 'y', 'ರ': 'r', 'ಲ': 'l', 'ವ': 'v', 'ಶ': 'ś', 'ಷ': 'ṣ', 'ಸ': 's', 'ಹ': 'h', 'ಳ': 'ḷ',
        'ಱ': 'ṟ', 'ೞ': 'ḻ'  # Obsolete but included for completeness
    }
    KANNADA_VOWELS = {
        'ಅ': 'a', 'ಆ': 'ā', 'ಇ': 'i', 'ಈ': 'ī', 'ಉ': 'u', 'ಊ': 'ū',
        'ಋ': 'ṛ', 'ಎ': 'e', 'ಏ': 'ē', 'ಐ': 'ai', 'ಒ': 'o', 'ಓ': 'ō', 'ಔ': 'au'
    }
    KANNADA_MATRAS = {
        'ಾ': 'ā', 'ಿ': 'i', 'ೀ': 'ī', 'ು': 'u', 'ೂ': 'ū',
        'ೃ': 'ṛ', 'ೆ': 'e', 'ೇ': 'ē', 'ೈ': 'ai', 'ೊ': 'o', 'ೋ': 'ō', 'ೌ': 'au'
    }
    KANNADA_MODIFIERS = {'ಂ': 'ṃ', 'ಃ': 'ḥ'}
    VIRAMA = '್'
    ZWNJ = '\u200C'
    ZWJ = '\u200D'

    result = ""
    for i, char in enumerate(text):
        if char in KANNADA_CONSONANTS:
            result += KANNADA_CONSONANTS[char]
            # Lookahead to see if we need to append the inherent 'a' (schwa)
            if i + 1 < len(text):
                next_char = text[i + 1]
                if next_char not in KANNADA_MATRAS and next_char != VIRAMA:
                    result += 'a'
            else:
                result += 'a'  # End of string gets the inherent 'a'
        elif char in KANNADA_VOWELS:
            result += KANNADA_VOWELS[char]
        elif char in KANNADA_MATRAS:
            result += KANNADA_MATRAS[char]
        elif char == VIRAMA:
            pass  # We simply skip it; the lookahead logic already stopped the 'a' from being appended
        elif char in KANNADA_MODIFIERS:
            result += KANNADA_MODIFIERS[char]
        elif char in [ZWNJ, ZWJ]:
            pass  # Ignore invisible joiners
        else:
            result += char  # Passes through spaces, english letters, and punctuation

    return result


# --- NEW MICRO-PROMPTS ---

DRAFTER_PROMPT = """
You are an expert Kannada linguist. 
TASK: Write ONE brief (<= 8 words), formal example sentence in Kannada using the provided target word, and provide its English translation.

CRITICAL RESTRICTION: Do NOT output Wikitext formatting. Do NOT output conversational filler.
You MUST output exactly in this format:
KANNADA: [Your Kannada sentence here]
ENGLISH: [Your English translation here]
"""

LOGIC_AUDITOR_PROMPT = """
You are a strict Linguistic QA Editor for Kannada. 
TASK: Review the provided Kannada sentence and its English translation. 

RULES:
1. PRESERVATION FIRST: If the drafted Kannada sentence is already grammatically flawless, formal, and naturally uses the target word, YOU MUST KEEP IT EXACTLY AS IS. Do not rewrite or alter a sentence just for the sake of making a change.
2. GRAMMAR AUDIT: Only make corrections if there is a genuine error. If you must correct it, pay strict attention to Kannada subject-verb agreement (e.g., ensure gender and number match perfectly, such as 'ಅವಳು' with a feminine verb ending, not a neuter one).
3. TRANSLATION MATCH: Ensure the English translation accurately reflects the Kannada sentence.

CRITICAL RESTRICTION: Do NOT output Wikitext. Do NOT output conversational filler.
You MUST output exactly in this format:
KANNADA: [Final Kannada sentence here]
ENGLISH: [Final English translation here]
"""

# --- APP UI ---
st.set_page_config(page_title="Kannada Wiktionary Gen", page_icon="🌿")

DRAFTER_MODEL = 'translategemma:27b'
LOGIC_MODEL = 'translategemma:27b'

st.title("Kannada Wiktionary Generator")
word = st.text_input("Enter word:")
translation = st.text_input("Enter translation:")
pos_categories = st.multiselect("POS:", ["Noun", "Verb", "Adjective", "Adverb"], default=["Noun"])

if word:
    ground_truth = load_ground_truth()
    if word in ground_truth:
        st.success("Found in Ground Truth!")
        st.text_area("Entry:", ground_truth[word], height=400)
    else:
        if st.button("Generate Wikitext"):
            st.divider()
            st.markdown("### 🔍 Generation Audit Trail")

            try:
                start_time = time.time()

                # --- PYTHON PRE-COMPUTATION ---
                morphology_block = get_dynamic_morphology_block(word, pos_categories)
                if "IRREGULAR_CHECK" in morphology_block:
                    stem = word.removesuffix("ು")
                    morphology_block = morphology_block.replace("IRREGULAR_CHECK",
                                                                f"{{{{kn-conj-u|{word}|{stem}ಿ|{stem}ಿದ}}}}")

                primary_pos = pos_categories[0] if pos_categories else "Noun"
                pos_template = f"{{{{kn-{primary_pos.lower()}}}}}"
                formatted_translation = ", ".join([f"[[{t.strip()}]]" for t in translation.split(",") if t.strip()])

                sanskrit_triggers = set("ಖಘಛಝಠಢಥಧಫಭಋೃಶಷಃಙಞ")
                etymology_line = "{{bor|kn|sa|DEVANAGARI_WORD}}" if any(
                    char in word for char in sanskrit_triggers) else "{{rfe|kn}}"

                with st.expander("⚙️ Python Deterministic Logic", expanded=False):
                    st.write(f"**Etymology:** `{etymology_line}`")
                    st.write(f"**Morphology Template:** `{morphology_block}`")
                    st.write(f"**Definitions:** `{formatted_translation}`")

                examples_block = get_few_shot_sentences(ground_truth, word)

                # --- STEP 1: DRAFTING (27B) ---
                with st.expander("📝 Step 1: Drafter (27B)", expanded=True):
                    step1_placeholder = st.empty()
                    drafter_user_content = f"### Context Examples:\n{examples_block}\n\n### Target Word: {word}\nMeaning: {translation}\n"
                    draft_output = ""
                    for chunk in ollama.chat(model=DRAFTER_MODEL,
                                             messages=[{'role': 'system', 'content': DRAFTER_PROMPT},
                                                       {'role': 'user', 'content': drafter_user_content}], stream=True):
                        draft_output += chunk['message']['content']
                        step1_placeholder.markdown(draft_output)

                    # Parse the sentence and translation
                    kn_draft, en_draft = parse_kannada_english(draft_output)

                # --- STEP 2: LOGIC AUDIT (27B) ---
                with st.expander("🛡️ Step 2: Logic Auditor (27B)", expanded=True):
                    step2_placeholder = st.empty()
                    logic_user_content = f"Target Word: {word}\nRequired Meaning: {translation}\n\nDraft:\nKANNADA: {kn_draft}\nENGLISH: {en_draft}"
                    logic_output = ""
                    for chunk in ollama.chat(model=LOGIC_MODEL,
                                             messages=[{'role': 'system', 'content': LOGIC_AUDITOR_PROMPT},
                                                       {'role': 'user', 'content': logic_user_content}], stream=True):
                        logic_output += chunk['message']['content']
                        step2_placeholder.markdown(logic_output)

                    # Parse the finalized sentence and translation
                    final_kn, final_en = parse_kannada_english(logic_output)
                    # Failsafe if the parser fails
                    if not final_kn: final_kn, final_en = kn_draft, en_draft

                # --- STEP 3: TRANSLITERATION (PYTHON) ---
                with st.expander("🔤 Step 3: Transliteration Proofreader (Python Logic)", expanded=True):
                    final_tr = transliterate_kannada_to_iso(final_kn)
                    st.markdown(f"**Target Sentence:** `{final_kn}`")
                    st.markdown(f"**ISO 15919 Transliteration:** `{final_tr}`")

                # --- PYTHON FINAL ASSEMBLY ---
                st.success(f"Generation Complete! ({format_time(time.time() - start_time)})")

                final_wikitext = f"""==Kannada==

===Etymology===
{etymology_line}

===Pronunciation===
* {{{{kn-IPA|{word}}}}}

==={primary_pos}===
{pos_template}
# {formatted_translation}

====Usage notes====
* {{{{ux|kn|{final_kn}|tr={final_tr}|t={final_en}}}}}

{morphology_block}

===References===
* {{{{R:kn:Alar}}}}"""

                st.session_state['current_result'] = final_wikitext

            except Exception as e:
                st.error(f"Pipeline Error: {e}")

    if 'current_result' in st.session_state:
        st.divider()
        st.markdown("### ✨ Final Wiktionary Entry")
        edited_entry = st.text_area("Verify & Edit:", st.session_state['current_result'], height=400)
        if st.button("💾 Save to Database"):
            save_to_ground_truth(word, edited_entry)
            st.success("Entry saved successfully!")
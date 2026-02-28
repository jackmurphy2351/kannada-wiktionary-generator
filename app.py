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
    """Parses existing ground truth to extract ONLY the example sentences for the Drafter."""
    if not current_ground_truth: return ""
    examples = []
    for word, wikitext in current_ground_truth.items():
        if len(examples) >= count: break
        # Regex to extract the Kannada and English parts from the ux template
        match = re.search(r'\{\{ux\|kn\|(.*?)\|tr=.*?\|t=(.*?)\}\}', wikitext)
        if match:
            kn_sent = match.group(1).strip()
            en_sent = match.group(2).strip()
            examples.append(f"Target Word: {word}\nKANNADA: {kn_sent}\nENGLISH: {en_sent}\n---")
    return "\n".join(examples)


def parse_kannada_english(text):
    """Safely extracts the Kannada and English strings from LLM output."""
    kn, en = "", ""
    if "KANNADA:" in text and "ENGLISH:" in text:
        kn = text.split("KANNADA:")[1].split("ENGLISH:")[0].replace("```", "").strip()
        en = text.split("ENGLISH:")[1].replace("```", "").strip()
    return kn, en


# --- NEW MICRO-PROMPTS ---

DRAFTER_PROMPT = """
You are an expert Kannada linguist. 
TASK: Write ONE sophisticated, formal example sentence in Kannada using the provided target word, and provide its English translation.

CRITICAL RESTRICTION: Do NOT output Wikitext formatting. Do NOT output conversational filler.
You MUST output exactly in this format:
KANNADA: [Your Kannada sentence here]
ENGLISH: [Your English translation here]
"""

LOGIC_AUDITOR_PROMPT = """
You are a strict Linguistic QA Editor. 
TASK: Review the provided Kannada sentence and its English translation. 
1. Ensure the Kannada sentence is grammatically flawless, formal, and naturally uses the target word.
2. Ensure the English translation is accurate.

CRITICAL RESTRICTION: Do NOT output Wikitext. Do NOT output conversational filler.
You MUST output exactly in this format:
KANNADA: [Corrected Kannada sentence here]
ENGLISH: [Corrected English translation here]
"""

TRANSLIT_AUDITOR_PROMPT = """
TASK: Provide the exact ISO 15919 Roman transliteration for the provided Kannada text.

RULES:
1. CHARACTER-LITERAL MAPPING: Map strictly character-by-character.
2. ASPIRATED CONSONANTS (MAHAPRANA): Do NOT drop the 'h' in aspirated letters. 
   - ಭ = bh (e.g., ಸಮಾರಂಭದಲ್ಲಿ is samāraṃbhadalli, NEVER samāraṃbadalli).
   - ಘ = gh, ಛ = ch, ಠ = ṭh, ಢ = ḍh, ಥ = th, ಧ = dh, ಫ = ph.
3. GEMINATION (VATTAKSHARA): When a consonant has a subscript of itself, you MUST double the Latin consonant. 
   - ದ್ದ = dd (e.g., ಧರಿಸಿದ್ದರು is dharisiddaru, NEVER dharisidaru).
   - ಕ್ಕ = kk, ಟ್ಟ = ṭṭ, ನ್ನ = nn, ಮ್ಮ = mm, ಲ್ಲ = ll, etc.
4. VOWEL SIGN (MATRA) MAPPING: Ensure the transliteration follows the vowel sign attached to the consonant:
   - Inherited (no sign): a (e.g., ತ = ta)
   - ಾ = ā | ಿ = i | ೀ = ī | ು = u | ೂ = ū
   - ೆ = e | ೇ = ē | ೊ = o | ೋ = ō
   - ೌ = au | ೈ = ai
5. NO SCHWA DELETION: Do NOT drop the inherent 'a' at the end of words (e.g., ಔಪಚಾರಿಕ is aupacārika).
6. ANUSVARA: ಂ = ṃ.
7. SANDHI PRESERVATION: If the script has no spaces, the transliteration must have no spaces.
8. CASE: Do not arbitrarily capitalize words in the middle of sentences.

CRITICAL: Output ONLY the raw transliterated string. No labels, no markdown, no quotes.
"""

# --- APP UI ---
st.set_page_config(page_title="Kannada Wiktionary Gen", page_icon="🌿")

DRAFTER_MODEL = 'translategemma:27b'
LOGIC_MODEL = 'gemma2:9b'
PROOFREADER_MODEL = 'translategemma:12b'

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

                # --- STEP 2: LOGIC AUDIT (9B) ---
                with st.expander("🛡️ Step 2: Logic Auditor (9B)", expanded=True):
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

                # --- STEP 3: TRANSLITERATION (12B) ---
                with st.expander("🔤 Step 3: Transliteration Proofreader (12B)", expanded=True):
                    step3_placeholder = st.empty()
                    translit_output = ""
                    for chunk in ollama.chat(model=PROOFREADER_MODEL,
                                             messages=[{'role': 'system', 'content': TRANSLIT_AUDITOR_PROMPT},
                                                       {'role': 'user', 'content': final_kn}], stream=True):
                        translit_output += chunk['message']['content']
                        step3_placeholder.markdown(translit_output)

                    final_tr = translit_output.replace("```", "").strip()

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
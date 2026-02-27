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
    """Loads the knowledge base from JSON."""
    if os.path.exists(JSON_FILE) and os.path.getsize(JSON_FILE) > 0:
        with open(JSON_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}

def save_to_ground_truth(word, entry):
    """Saves a verified wikitext entry to the JSON file."""
    data = load_ground_truth()
    data[word] = entry
    with open(JSON_FILE, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

def format_time(seconds):
    """Formats seconds into H:M:S or M:S or S based on duration."""
    seconds = int(seconds)
    if seconds < 60:
        return f"{seconds}s"
    minutes = seconds // 60
    remaining_seconds = seconds % 60
    if minutes < 60:
        return f"{minutes}m {remaining_seconds}s"
    return f"{minutes // 60}h {minutes % 60}m {remaining_seconds}s"

def get_template_logic(word, pos_categories):
    """
    Returns the correct Wiktionary template strings based on word endings.
    """
    instructions = []
    last_char = word[-1] if word else ""

    for pos in pos_categories:
        if pos == "Noun":
            if last_char == "ು":
                stem = word.removesuffix("ು")
                instructions.append(f"Noun: {{{{kn-decl-u|{word}|{stem}}}}}")
            elif last_char in {"ಿ", "ೆ", "ೈ"}:
                instructions.append(f"Noun: {{{{kn-decl-e-i-ai|{word}|{word}}}}}")
            else:
                instructions.append(f"Noun: {{{{kn-decl-a|{word}}}}}")

        elif pos == "Verb":
            if word.endswith("ಕೊಳ್ಳು"):
                prefix = word.removesuffix("ಕೊಳ್ಳು")
                instructions.append(f"Verb: {{{{kn-conj-koḷḷu|{prefix}}}}}")
            elif word.endswith("ಿಸು"):
                stem = word.removesuffix("ು")
                instructions.append(f"Verb: {{{{kn-conj-isu|{word}|{stem}}}}}")
            elif last_char in {"ಿ", "ೆ", "ೈ"}:
                instructions.append(f"Verb: {{{{kn-conj-e-i-other|{word}|{word}ಯ|{word}ದ|{word}}}}}")
            else:
                instructions.append("Verb: IRREGULAR_CHECK")

    return " AND ".join(instructions) if instructions else "No specific morphology templates required."

def get_few_shot_examples(current_ground_truth, pos_categories, target_word, count=2):
    """
    Formats a subset of verified entries with a single-pass search optimization.
    """
    if not current_ground_truth:
        return ""

    primary_pos = pos_categories[0] if pos_categories else None
    structural_target = ""

    if primary_pos == "Verb":
        if target_word.endswith("ಿಸು"):
            structural_target = "kn-conj-isu"
        elif target_word.endswith("ಕೊಳ್ಳು"):
            structural_target = "kn-conj-koḷḷu"
        elif target_word[-1] in {"ಿ", "ೆ", "ೈ"}:
            structural_target = "kn-conj-e-i-other"
        else:
            structural_target = "kn-conj-u"
    elif primary_pos == "Noun":
        last_char = target_word[-1] if target_word else ""
        if last_char == "ು":
            structural_target = "kn-decl-u"
        elif last_char in {"ಿ", "ೆ", "ೈ"}:
            structural_target = "kn-decl-e-i-ai"
        else:
            structural_target = "kn-decl-a"

    examples = []
    fallback_entries = []

    for word, wikitext in current_ground_truth.items():
        if len(examples) >= count:
            break

        if any(f"==={pos}===" in wikitext for pos in pos_categories):
            if primary_pos and f"==={primary_pos}===" in wikitext and structural_target in wikitext:
                examples.append(f"Word: {word}\nOutput:\n{wikitext}\n---")
            elif len(fallback_entries) < count:
                fallback_entries.append(f"Word: {word}\nOutput:\n{wikitext}\n---")

    while len(examples) < count and fallback_entries:
        examples.append(fallback_entries.pop(0))

    return "\n".join([f"\nExample {i + 1}:\n{ex}" for i, ex in enumerate(examples)])

# --- SYSTEM PROMPTS ---

KANNADA_LEXICOGRAPHER_PROMPT = """
You are an expert Kannada Lexicographer and Dravidian Linguist. 
TASK: Output raw Wikitext ONLY. 

I. LINGUISTIC PRECISION & EXAMPLES:
- Use <examples> as the "Golden Standard". Pay exact attention to how they format example sentences.
- EXAMPLES REQUIRED: You MUST generate at least one natural, high-quality SOV Kannada example sentence for EACH meaning.
- STRICT UX TEMPLATE: Use {{ux|kn|KANNADA_SCRIPT|tr=TRANSLITERATION|t=ENGLISH_TRANSLATION}}.
- SCRIPT ENFORCEMENT: Example sentences MUST be 100% Kannada script. NEVER use Telugu, Tamil, or Devanagari characters.
- KANNADA-ONLY TEMPLATES: The templates {{kn-IPA|...}} and {{kn-decl-...|...}} MUST use the word in Kannada script, NEVER Latin/Roman script.

II. ETYMOLOGY & HALLUCINATION:
- NEVER guess etymologies. If uncertain, use {{rfe|kn}}.
- SANSKRIT LOAN INDICATORS: If a word contains aspirated consonants (ಖ, ಘ, ಛ, ಝ, ಠ, ಢ, ಥ, ಧ, ಫ, ಭ), it is a Sanskrit borrowing ({{bor|kn|sa|...}}). DO NOT claim it is native Dravidian.
- Example: 'ಭೂಮಿ' contains 'ಭ', so it is borrowed from Sanskrit.

III. ISO 15919 TRANSLITERATION TABLE:
Strictly follow this mapping for the `tr=` parameter:
- Vowels: ಆ=ā, ಈ=ī, ಊ=ū, ಏ=ē, ಓ=ō, ಐ=ai, ಔ=au
- Retroflex (Underdot): ಟ=ṭ, ಠ=ṭh, ಡ=ḍ, ಢ=ḍh, ಣ=ṇ, ಳ=ḷ, ಷ=ṣ
- Dental (NO Underdot): ತ=t, ಥ=th, ದ=d, ಧ=dh, ನ=n
- CRITICAL: Never use 'ṭ' for 'ತ'. Use 't'.
"""

VALIDATOR_PROMPT = """
You are a ruthless QA Editor for Kannada Wiktionary. 
Review the draft and perform these EXACT fixes:

1. TRANSLITERATION AUDIT (CRITICAL): 
   - Check every 'tr=' parameter.
   - If the Kannada letter is 'ತ' (dental), the transliteration MUST be 't'.
   - If the Kannada letter is 'ಟ' (retroflex), the transliteration MUST be 'ṭ'.
   - Fix missing macrons (ā, ī, ū, ē, ō) and underdots (ṭ, ḍ, ṇ, ḷ, ṣ).

2. TEMPLATE SCRIPT CHECK:
   - Ensure {{kn-IPA|...}} and {{kn-decl-...|...}} contain ONLY Kannada script. If you see Latin characters like 'bhūmi', replace them with the Kannada script 'ಭೂಮಿ'.

3. SCRIPT CLEANING:
   - Replace any accidentally used Telugu or Devanagari characters with Kannada equivalents.

4. ETYMOLOGY CHECK:
   - If a word with aspirated letters (like ಭ) is labeled as 'inh' from 'dra-pro', change it to 'bor' from 'sa'.

CRITICAL RULE: Return ONLY the corrected raw Wikitext. DO NOT output explanations, conversational filler, or markdown blocks. Your output MUST start exactly with ==Kannada==.
"""

# --- APP UI ---
st.set_page_config(page_title="Kannada Wiktionary Gen", page_icon="🌿")

SELECTED_MODEL = 'translategemma:27b'
VALIDATOR_MODEL = 'gemma2:9b'

st.sidebar.title("Model Information")
st.sidebar.info(f"Drafting: `{SELECTED_MODEL}`\nValidating: `{VALIDATOR_MODEL}`")

st.title("Kannada Wiktionary Generator")

word = st.text_input("Enter a Kannada word:")
translation = st.text_input("Enter the English translation:")
st.caption("💡 **Tip:** Use infinitive forms for verbs (e.g., *to walk*).")

if "last_word" not in st.session_state or st.session_state["last_word"] != word:
    st.session_state["last_word"] = word
    st.session_state.pop('current_result', None)

pos_categories = st.multiselect("Select Parts of Speech:", ["Noun", "Verb", "Adjective", "Adverb", "Postposition"], default=["Noun"])

if word:
    ground_truth = load_ground_truth()

    if word in ground_truth:
        st.success("Found in Ground Truth!")
        st.text_area("Wiktionary Entry:", ground_truth[word], height=400)
    else:
        if st.button("Generate Wikitext"):
            progress_bar = st.progress(0)
            status_text = st.empty()
            timer_text = st.empty()

            try:
                start_time = time.time()
                template_instruction = get_template_logic(word, pos_categories)

                if template_instruction == "IRREGULAR_CHECK":
                    stem = word.removesuffix("ು")
                    template_instruction = f"{{{{kn-conj-u|{word}|{stem}ಿ|{stem}ಿದ}}}}"

                examples_block = get_few_shot_examples(ground_truth, pos_categories, word, count=2)

                full_prompt = (
                    f"### Context Examples:\n{examples_block}\n"
                    f"### Target Task:\n"
                    f"Word: {word}\n"
                    f"Primary Meaning: {translation}\n"
                    f"Morphology Template: {template_instruction}\n\n"
                    f"Generate the full Wiktionary entry for '{word}'. "
                )

                status_text.text(f"Step 1/2: Drafting with {SELECTED_MODEL}...")
                progress_bar.progress(25)

                draft_content = ""
                for chunk in ollama.chat(model=SELECTED_MODEL, messages=[
                    {'role': 'system', 'content': KANNADA_LEXICOGRAPHER_PROMPT},
                    {'role': 'user', 'content': full_prompt}
                ], stream=True):
                    draft_content += chunk['message']['content']
                    timer_text.markdown(f"**⏱️ Elapsed Time:** `{format_time(time.time() - start_time)}`")

                status_text.text(f"Step 2/2: Auditing with {VALIDATOR_MODEL}...")
                progress_bar.progress(60)

                final_content = ""
                validation_prompt = f"Target Word: {word}\n\nDraft Wikitext to Correct:\n{draft_content}"

                for chunk in ollama.chat(model=VALIDATOR_MODEL, messages=[
                    {'role': 'system', 'content': VALIDATOR_PROMPT},
                    {'role': 'user', 'content': validation_prompt}
                ], stream=True):
                    final_content += chunk['message']['content']
                    timer_text.markdown(f"**⏱️ Elapsed Time:** `{format_time(time.time() - start_time)}`")

                progress_bar.progress(100)
                status_text.text("Done!")

                # Remove any remaining reasoning or conversational text
                if "==Kannada==" in final_content:
                    final_content = "==Kannada==" + final_content.split("==Kannada==")[-1]

                st.session_state['current_result'] = final_content

            except Exception as e:
                st.error(f"Error: {e}")
            finally:
                time.sleep(1)
                progress_bar.empty()
                status_text.empty()
                timer_text.empty()

    if 'current_result' in st.session_state:
        st.subheader("Edit & Verify")
        edited_entry = st.text_area("Final Wikitext:", st.session_state['current_result'], height=400)

        st.markdown("---")
        if st.button("Save to Ground Truth"):
            save_to_ground_truth(word, edited_entry)
            st.balloons()
            st.success(f"Verified entry for '{word}' added!")
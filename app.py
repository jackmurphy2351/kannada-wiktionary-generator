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
    hours = minutes // 60
    remaining_minutes = minutes % 60
    return f"{hours}h {remaining_minutes}m {remaining_seconds}s"


def get_template_logic(word, pos_categories):
    """
    Returns the correct Wiktionary template strings based on word endings.
    Now accepts a LIST of parts of speech (pos_categories).
    """
    instructions = []
    last_char = word[-1] if word else ""

    for pos in pos_categories:
        if pos == "Noun":
            if last_char == "ು":
                stem = word.removesuffix("ು")
                instructions.append(f"Noun: {{{{kn-decl-u|{word}|{stem}}}}}")
            elif last_char in ["ಿ", "ೆ", "ೈ"]:
                instructions.append(f"Noun: {{{{kn-decl-e-i-ai|{word}|{word}}}}}")
            else:
                instructions.append(f"Noun: {{{{kn-decl-a|{word}}}}}")

        elif pos == "Verb":
            if word.endswith("ಕೊಳ್ಳು"):
                prefix = word.removesuffix("ಕೊಳ್ಳು")
                instructions.append(f"Verb: {{{{kn-conj-koḷḷu|{prefix}}}}}")
            elif word.endswith("ಿಸು"):
                stem = word.removesuffix("ು")  # removes the final 'u' to get the stem
                instructions.append(f"Verb: {{{{kn-conj-isu|{word}|{stem}}}}}")
            elif last_char in ["ಿ", "ೆ", "ೈ"]:
                instructions.append(f"Verb: {{{{kn-conj-e-i-other|{word}|{word}ಯ|{word}ದ|{word}}}}}")
            else:
                instructions.append("Verb: IRREGULAR_CHECK")

    # Join the instructions so the LLM knows what to do for each section
    return " AND ".join(instructions) if instructions else "No specific morphology templates required."


def get_few_shot_examples(current_ground_truth, pos_categories, target_word, count=4):
    """
    Formats a subset of verified entries. Prioritizes entries that match
    AT LEAST ONE of the target's parts of speech.
    """
    examples = ""
    matched_entries = []

    # Let's try to match the structural template of the FIRST selected POS
    primary_pos = pos_categories[0] if pos_categories else None
    structural_target = ""

    if primary_pos == "Verb":
        if target_word.endswith("ಿಸು"):
            structural_target = "kn-conj-isu"
        elif target_word.endswith("ಕೊಳ್ಳು"):
            structural_target = "kn-conj-koḷḷu"
        elif target_word[-1] in ["ಿ", "ೆ", "ೈ"]:
            structural_target = "kn-conj-e-i-other"
        else:
            structural_target = "kn-conj-u"
    elif primary_pos == "Noun":
        last_char = target_word[-1] if target_word else ""
        if last_char == "ು":
            structural_target = "kn-decl-u"
        elif last_char in ["ಿ", "ೆ", "ೈ"]:
            structural_target = "kn-decl-e-i-ai"
        else:
            structural_target = "kn-decl-a"

    # Pass 1: Match POS AND Structural Template
    if primary_pos:
        for word, wikitext in current_ground_truth.items():
            if f"==={primary_pos}===" in wikitext and structural_target in wikitext:
                matched_entries.append((word, wikitext))
            if len(matched_entries) == count: break

    # Pass 2: Fill with ANY matches to the requested POS categories
    if len(matched_entries) < count:
        for word, wikitext in current_ground_truth.items():
            # Check if any of the requested POS headers are in this wikitext
            if any(f"==={pos}===" in wikitext for pos in pos_categories) and (word, wikitext) not in matched_entries:
                matched_entries.append((word, wikitext))
            if len(matched_entries) == count: break

    # Formatting the output for the prompt
    for i, (word, wikitext) in enumerate(matched_entries):
        examples += f"\nExample {i + 1}:\nWord: {word}\nOutput:\n{wikitext}\n---\n"

    return examples


# --- SYSTEM PROMPTS ---

# Unified primary prompt for TranslateGemma (12B)
KANNADA_LEXICOGRAPHER_PROMPT = """
You are an expert Kannada Lexicographer and Dravidian Linguist. 
TASK: Output raw Wikitext ONLY. 

I. LINGUISTIC PRECISION:
- Use <examples> as the "Golden Standard" for structure.
- GLIDE RULE: For verbs ending in -ೆ (like ನಡೆ), you MUST use the 'ಯ' (ya) glide before vowel-starting suffixes (e.g., ನಡೆಯಬೇಕು, ನಡೆಯುತ್ತಿದೆ).
- CAUSATIVE WARNING: Do not confuse base verbs with causatives (e.g., use 'ನಡೆ' for walking, NOT 'ನಡೆಸು' which means to conduct).
- VERB SUBJECTS: Ensure verbs match subjects (e.g., -ಳು for 'ಅವಳು', -ನು for 'ಅವನು').

II. ANTI-HALLUCINATION & ETYMOLOGY:
- NEVER guess etymologies. If uncertain, use exactly: {{rfe|kn}}.
- Distinguish between 'tatsama' (Sanskrit loans: {{bor|kn|sa|...}}) and native words ({{inh|kn|dra-pro|*...}}).
- Words starting with ಪ್ರ-, ವಿ-, or ಸಂ- are typically Sanskrit.

III. ISO 15919 TRANSLITERATION TABLE:
Strictly follow this mapping for the `tr=` parameter:
- Vowels: ಆ=ā, ಈ=ī, ಊ=ū, ಏ=ē, ಓ=ō, ಐ=ai, ಔ=au
- Retroflex: ಟ=ṭ, ಠ=ṭh, ಡ=ḍ, ಢ=ḍh, ಣ=ṇ, ಳ=ḷ
- Dental: ತ=t, ಥ=th, ದ=d, ಧ=dh, ನ=n
- Palatal: ಚ=c (NOT ch), ಛ=ch
- Aspirated: ಖ=kh, ಘ=gh, ಛ=ch, ಝ=jh, ಥ=th, ಧ=dh, ಫ=ph, ಭ=bh
"""

# Hardened Validator for secondary pass
VALIDATOR_PROMPT = """
You are a ruthless QA Editor for Kannada Wiktionary. 
Review the draft and perform these EXACT fixes:

1. TRANSLITERATION AUDIT: 
   - Cross-check EVERY `tr=` value against the Kannada script in the same line.
   - Fix missing underdots for retroflexes: ಟ, ಡ, ಣ, ಳ → ṭ, ḍ, ṇ, ḷ.
   - Fix missing macrons for long vowels: ಆ, ಈ, ಊ, ಏ, ಓ → ā, ī, ū, ē, ō.
   - Change 'ch' to 'c' for ಚ.
   - Example: Change tr=Avalu to tr=Avaḷu. Change tr=chennagi to tr=cennāgi.

2. ETYMOLOGY VETTING:
   - If the etymology claims a native Dravidian word (like ನಡೆ) comes from English or looks like a hallucination, replace it with {{rfe|kn}}.

3. GLIDE CHECK:
   - Ensure verbs ending in -ೆ or -ಿ use the 'y' glide in transliteration when applicable (e.g., naḍeyabēku).

Return ONLY the corrected raw Wikitext. NEVER wrap the wikitext in markdown code blocks like "```wiktionary".
"""

# --- APP UI ---
st.set_page_config(page_title="Kannada Wiktionary Gen", page_icon="🌿")

st.sidebar.title("Model Settings")
model_choice = st.sidebar.radio(
    "Choose Brain Size:",
    options=["TranslateGemma (Fast - 12B)", "Sarvam-M (Deep - 24B)"],
    index=0
)

SELECTED_MODEL = 'translategemma:12b' if "Fast" in model_choice else 'mashriram/sarvam-m:latest'
CURRENT_SYSTEM_PROMPT = KANNADA_LEXICOGRAPHER_PROMPT if "Fast" in model_choice else DEEP_PROMPT
st.sidebar.info(f"Using: `{SELECTED_MODEL}`")

st.title("Kannada Wiktionary Generator")

word = st.text_input("Enter a Kannada word:")

# UI Upgrade: Added formatting instructions for the translation field
translation = st.text_input("Enter the English translation:")
st.caption(
    "💡 **Tip for better results:** Separate multiple meanings with commas (e.g., *beginning, start*). For verbs, always use the infinitive form (e.g., *to use, to employ*).")

if "last_word" not in st.session_state or st.session_state["last_word"] != word:
    st.session_state["last_word"] = word
    for key in ['current_result', 'sandbox_results']:
        if key in st.session_state:
            del st.session_state[key]

pos_categories = st.multiselect("Select Parts of Speech:", ["Noun", "Verb", "Adjective", "Adverb", "Postposition"],
                                default=["Noun"])

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

                # --- SETUP PROMPT ---
                template_instruction = get_template_logic(word, pos_categories)

                if template_instruction == "IRREGULAR_CHECK":
                    if "Fast" in model_choice:
                        stem = word[:-1]
                        template_instruction = f"{{{{kn-conj-u|{stem}|{stem}ಿ|{stem}ಿದ}}}}"
                    else:
                        template_instruction = "Use {{kn-conj-u-irreg}} for geminates or {{kn-conj-u}} for regular forms."

                example_count = 3 if "Fast" in model_choice else 4
                examples_block = get_few_shot_examples(ground_truth, pos_categories, word, count=example_count)

                full_prompt = (
                    f"### Context Examples:\n{examples_block}\n"
                    f"### Target Task:\n"
                    f"Word: {word}\n"
                    f"Primary Meaning: {translation}\n"
                    f"Morphology Template: {template_instruction}\n\n"
                    f"Generate the full Wiktionary entry for '{word}'. "
                    f"Ensure etymology is linguistically plausible for a Dravidian language."
                )

                # --- STEP 1: GENERATE DRAFT ---
                status_text.text(f"Step 1/2: Drafting entry with {model_choice}...")
                progress_bar.progress(25)

                draft_content = ""
                for chunk in ollama.chat(model=SELECTED_MODEL, messages=[
                    {'role': 'system', 'content': CURRENT_SYSTEM_PROMPT},
                    {'role': 'user', 'content': full_prompt}
                ], stream=True):
                    draft_content += chunk['message']['content']
                    elapsed = time.time() - start_time
                    timer_text.markdown(f"**⏱️ Elapsed Time:** `{format_time(elapsed)}`")

                if "</think>" in draft_content:
                    draft_content = draft_content.split("</think>")[-1].strip()

                # --- STEP 2: VALIDATE AND CORRECT ---
                status_text.text("Step 2/2: Validating transliteration & etymology...")
                progress_bar.progress(60)

                final_content = ""
                validation_prompt = f"Target Word: {word}\n\nDraft Wikitext to Correct:\n{draft_content}"

                for chunk in ollama.chat(model=SELECTED_MODEL, messages=[
                    {'role': 'system', 'content': VALIDATOR_PROMPT},
                    {'role': 'user', 'content': validation_prompt}
                ], stream=True):
                    final_content += chunk['message']['content']
                    elapsed = time.time() - start_time
                    timer_text.markdown(f"**⏱️ Elapsed Time:** `{format_time(elapsed)}`")

                progress_bar.progress(100)
                status_text.text("Done!")

                if "</think>" in final_content:
                    final_content = final_content.split("</think>")[-1].strip()

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

        if st.button("Generate 3 Example Sentences"):
            progress_bar = st.progress(0)
            status_text = st.empty()
            timer_text = st.empty()

            try:
                start_time = time.time()
                status_text.text("Writing sentences...")
                progress_bar.progress(30)

                sandbox_prompt = f"Word: {word}\nMeaning: {translation}\nTask: Generate 3 simple SOV Kannada sentences using {{ux|kn|Kannada|t=English}}"

                full_content = ""
                for chunk in ollama.chat(model=SELECTED_MODEL, messages=[
                    {'role': 'system', 'content': 'Output raw Wikitext ONLY. Start with the first sentence.'},
                    {'role': 'user', 'content': sandbox_prompt}
                ], stream=True):
                    full_content += chunk['message']['content']
                    elapsed = time.time() - start_time
                    timer_text.markdown(f"**⏱️ Elapsed Time:** `{format_time(elapsed)}`")

                progress_bar.progress(100)
                status_text.text("Complete!")

                if "</think>" in full_content:
                    full_content = full_content.split("</think>")[-1].strip()
                st.session_state['sandbox_results'] = full_content
            except Exception as e:
                st.error(f"Error: {e}")
            finally:
                time.sleep(1)
                progress_bar.empty()
                status_text.empty()
                timer_text.empty()

        if 'sandbox_results' in st.session_state:
            st.code(st.session_state['sandbox_results'], language='text')

        st.markdown("---")
        if st.button("Save to Ground Truth"):
            save_to_ground_truth(word, edited_entry)
            st.balloons()
            st.success(f"Verified entry for '{word}' added!")
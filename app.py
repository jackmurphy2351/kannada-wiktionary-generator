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

# Optimized for TranslateGemma (12B): Focuses on cross-lingual accuracy and template adherence
FAST_PROMPT = """
You are a Kannada Lexicographer specializing in Wiktionary formatting. 
TASK: Output raw Wikitext ONLY. 

I. FEW-SHOT UTILIZATION:
- Analyze the provided <examples> for formatting patterns.
- Use the examples to determine how to structure the 'Etymology', 'Pronunciation', and 'Usage' sections for the target word.

II. ETYMOLOGY & SCRIPT:
- NEVER assume an English origin for native words.
- Use native scripts for cognates: Tamil (ta), Telugu (te), Malayalam (ml).
- If a word starts with ಪ್ರ-, ವಿ-, or ಸಂ-, it is likely a Sanskrit loan ({{bor|kn|sa|...}}).

III. TEMPLATE RULES:
- ALWAYS use {{ux|kn|Kannada|t=English}} for examples. 
- Ensure Roman transliteration (tr=) in {{ux}} is phonetically accurate (e.g., use 'd' for ದ, not 'ḍ').
"""

# Optimized for Sarvam-M (24B): Focuses on deep linguistic reasoning and etymological nuance
DEEP_PROMPT = """
You are an expert Kannada Lexicographer specializing in Dravidian linguistics. 
Output raw Wikitext ONLY. 

I. FEW-SHOT REASONING:
- Use the <examples> as a structural "Golden Standard".
- If an example shows a specific conjugation class (like kn-conj-isu), apply that logic to the target word if applicable.

II. LINGUISTIC DEPTH:
- Distinguish between 'tatsama' (direct Sanskrit loans) and 'tadbhava' (modified loans).
- For etymology, provide the Proto-Dravidian root (e.g., {{inh|kn|dra-pro|*...}}) if the word is native.
- Provide cognates with native scripts and meanings (t=meaning).

III. USAGE & GRAMMAR:
- Generate 2 natural, simple sentences.
- Ensure verbs end in the appropriate ending for their subject, e.g. -ಳು if the subject is "ಅವಳು."
- Include 'Usage notes' if the word has subtle semantic differences from English equivalents.
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
CURRENT_SYSTEM_PROMPT = FAST_PROMPT if "Fast" in model_choice else DEEP_PROMPT
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
            timer_text = st.empty()  # Placeholder for the counter

            try:
                start_time = time.time()  # Start the clock

                # BUG FIX: Swapped 'pos_category' to 'pos_categories'
                template_instruction = get_template_logic(word, pos_categories)

                if template_instruction == "IRREGULAR_CHECK":
                    if "Fast" in model_choice:
                        stem = word[:-1]
                        template_instruction = f"{{{{kn-conj-u|{stem}|{stem}ಿ|{stem}ಿದ}}}}"
                    else:
                        template_instruction = "Use {{kn-conj-u-irreg}} for geminates or {{kn-conj-u}} for regular forms."

                example_count = 3 if "Fast" in model_choice else 4

                # BUG FIX: Swapped 'pos_category' to 'pos_categories'
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

                status_text.text(f"Generating with {model_choice}...")
                progress_bar.progress(25)

                # Use streaming to update the timer during the long wait
                full_content = ""
                for chunk in ollama.chat(model=SELECTED_MODEL, messages=[
                    {'role': 'system', 'content': CURRENT_SYSTEM_PROMPT},
                    {'role': 'user', 'content': full_prompt}
                ], stream=True):
                    full_content += chunk['message']['content']
                    elapsed = time.time() - start_time
                    timer_text.markdown(f"**⏱️ Elapsed Time:** `{format_time(elapsed)}`")

                progress_bar.progress(100)
                status_text.text("Done!")

                if "</think>" in full_content:
                    full_content = full_content.split("</think>")[-1].strip()
                st.session_state['current_result'] = full_content

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
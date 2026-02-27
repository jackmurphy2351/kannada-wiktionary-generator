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


def get_few_shot_examples(current_ground_truth, pos_category, target_word, count=4):
    """Formats a subset of verified entries structurally matched to the target word."""
    examples = ""
    matched_entries = []

    structural_target = ""
    if pos_category == "Verb":
        if target_word.endswith("ಿಸು"):
            structural_target = "kn-conj-isu"
        elif target_word.endswith("ಕೊಳ್ಳು"):
            structural_target = "kn-conj-koḷḷu"
        elif target_word[-1] in ["ಿ", "ೆ", "ೈ"]:
            structural_target = "kn-conj-e-i-other"
        else:
            structural_target = "kn-conj-u"

    elif pos_category == "Noun":
        last_char = target_word[-1]
        if last_char == "ು":
            structural_target = "kn-decl-u"
        elif last_char in ["ಿ", "ೆ", "ೈ"]:
            structural_target = "kn-decl-e-i-ai"
        else:
            structural_target = "kn-decl-a"

    for word, wikitext in current_ground_truth.items():
        if f"==={pos_category}===" in wikitext and structural_target in wikitext:
            matched_entries.append((word, wikitext))
        if len(matched_entries) == count:
            break

    if len(matched_entries) < count:
        for word, wikitext in current_ground_truth.items():
            if f"==={pos_category}===" in wikitext and (word, wikitext) not in matched_entries:
                matched_entries.append((word, wikitext))
            if len(matched_entries) == count:
                break

    for i, (word, wikitext) in enumerate(matched_entries):
        examples += f"\nExample {i + 1}:\nWord: {word}\nOutput:\n{wikitext}\n---\n"

    return examples


def get_template_logic(word, pos):
    """Returns the correct Wiktionary template string based on word endings."""
    last_char = word[-1]
    if pos == "Noun":
        if last_char == "ು":
            return f"{{{{kn-decl-u|{word}|{word[:-1]}}}}}"
        elif last_char in ["ಿ", "ೆ", "ೈ"]:
            return f"{{{{kn-decl-e-i-ai|{word}|{word}}}}}"
        else:
            return f"{{{{kn-decl-a|{word}}}}}"
    elif pos == "Verb":
        if word.endswith("ಕೊಳ್ಳು"):
            return f"{{{{kn-conj-koḷḷu|{word[:-5]}}}}}"
        if word.endswith("ಿಸು"):
            return f"{{{{kn-conj-isu|{word}|{word[:-1]}}}}}"
        if last_char in ["ಿ", "ೆ", "ೈ"]:
            return f"{{{{kn-conj-e-i-other|{word}|{word}ಯ|{word}ದ|{word}}}}}"
        return "IRREGULAR_CHECK"
    return "Template not found"


# --- SYSTEM PROMPTS ---

FAST_PROMPT = """
You are a Kannada Lexicographer.
TASK: Output raw Wikitext ONLY. 
STRICT RULE: Do NOT explain the examples. Do NOT provide commentary. 
STRICT RULE: Start your response with "==Kannada==" and end with "[[Category:Kannada verbs]]" or a reference. 
1. Use ONLY Kannada script for Kannada words. 
2. For Sanskrit, use ONLY Devanagari (e.g., {{bor|kn|sa|...}}). 
3. For Dravidian cognates, use native scripts ONLY (Tamil, Telugu, Malayalam). 
4. NEVER transliterate non-Kannada words into Kannada script. 
5. Keep sentences short (SOV order). No conversational filler.
"""

DEEP_PROMPT = """
You are an expert Kannada Lexicographer specializing in Dravidian linguistics. 
Output raw Wikitext ONLY. 
1. Provide a detailed etymology. Distinguish between 'tatsama' (direct Sanskrit loans) and 'tadbhava' (modified loans). 
2. For cognates, provide the correct native script and the English gloss (t=meaning). 
3. Generate 3 natural usage examples using correct grammar and everyday context. 
4. Ensure the Wiktionary templates used match the specific morphological requirements of the word.
"""

# --- APP UI ---
st.set_page_config(page_title="Kannada Wiktionary Gen", page_icon="🌿")

st.sidebar.title("Model Settings")
model_choice = st.sidebar.radio(
    "Choose Brain Size:",
    options=["Sarvam-1 (Fast - 2B)", "Sarvam-M (Deep - 24B)"],
    index=0
)

SELECTED_MODEL = 'mashriram/sarvam-1:latest' if "Fast" in model_choice else 'mashriram/sarvam-m:latest'
CURRENT_SYSTEM_PROMPT = FAST_PROMPT if "Fast" in model_choice else DEEP_PROMPT
st.sidebar.info(f"Using: `{SELECTED_MODEL}`")

st.title("Kannada Wiktionary Generator")

word = st.text_input("Enter a Kannada word:")
translation = st.text_input("Enter the English translation:")

if "last_word" not in st.session_state or st.session_state["last_word"] != word:
    st.session_state["last_word"] = word
    for key in ['current_result', 'sandbox_results']:
        if key in st.session_state:
            del st.session_state[key]

pos_category = st.selectbox("Select Part of Speech:", ["Noun", "Verb", "Adjective", "Adverb"])

if word:
    ground_truth = load_ground_truth()

    if word in ground_truth:
        st.success("Found in Ground Truth!")
        st.text_area("Wiktionary Entry:", ground_truth[word], height=400)
    else:
        if st.button("Generate Wikitext"):
            progress_bar = st.progress(0)
            status_text = st.empty()

            try:
                template_instruction = get_template_logic(word, pos_category)

                # Change 1: Deterministic Template for Fast Model
                if template_instruction == "IRREGULAR_CHECK":
                    if "Fast" in model_choice:
                        # Provide a concrete pattern for the 2B model to mimic
                        stem = word[:-1]
                        template_instruction = f"{{{{kn-conj-u|{stem}|{stem}ಿ|{stem}ಿದ}}}}"
                    else:
                        template_instruction = "Use {{kn-conj-u-irreg}} for geminates or {{kn-conj-u}} for regular forms."

                example_count = 2 if "Fast" in model_choice else 4
                examples_block = get_few_shot_examples(ground_truth, pos_category, word, count=example_count)

                # Change 2: Re-ordered Prompt (Last Word Rule)
                full_prompt = (
                    f"<examples>\n{examples_block}\n</examples>\n\n"
                    f"Word: {word}\n"
                    f"Translation: {translation}\n"
                    f"Template to use: {template_instruction}\n\n"
                    f"STRICT TASK: Generate ONLY the Wikitext entry for {word}. "
                    f"Begin immediately with '==Kannada=='."
                )

                status_text.text(f"Generating with {model_choice}...")
                progress_bar.progress(25)

                response = ollama.chat(model=SELECTED_MODEL, messages=[
                    {'role': 'system', 'content': CURRENT_SYSTEM_PROMPT},
                    {'role': 'user', 'content': full_prompt}
                ])

                progress_bar.progress(100)
                status_text.text("Done!")

                content = response['message']['content']
                if "</think>" in content:
                    content = content.split("</think>")[-1].strip()
                st.session_state['current_result'] = content

            except Exception as e:
                st.error(f"Error: {e}")
            finally:
                time.sleep(1)
                progress_bar.empty()
                status_text.empty()

    if 'current_result' in st.session_state:
        st.subheader("Edit & Verify")
        edited_entry = st.text_area("Final Wikitext:", st.session_state['current_result'], height=400)

        if st.button("Generate 3 Example Sentences"):
            progress_bar = st.progress(0)
            status_text = st.empty()

            try:
                status_text.text("Brainstorming sentences...")
                progress_bar.progress(30)

                sandbox_prompt = f"Word: {word}\nMeaning: {translation}\nTask: Generate 3 simple SOV Kannada sentences using {{ux|kn|Kannada|t=English}}"

                resp = ollama.chat(model=SELECTED_MODEL, messages=[
                    {'role': 'system', 'content': 'Output raw Wikitext only.'},
                    {'role': 'user', 'content': sandbox_prompt}
                ])

                progress_bar.progress(100)
                status_text.text("Complete!")

                content = resp['message']['content']
                if "</think>" in content:
                    content = content.split("</think>")[-1].strip()
                st.session_state['sandbox_results'] = content
            except Exception as e:
                st.error(f"Error: {e}")
            finally:
                time.sleep(1)
                progress_bar.empty()
                status_text.empty()

        if 'sandbox_results' in st.session_state:
            st.code(st.session_state['sandbox_results'], language='text')

        st.markdown("---")
        if st.button("Save to Ground Truth"):
            save_to_ground_truth(word, edited_entry)
            st.balloons()
            st.success(f"Verified entry for '{word}' added!")
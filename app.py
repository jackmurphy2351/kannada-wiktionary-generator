import streamlit as st
import json
import ollama
import os
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


def get_few_shot_examples(current_ground_truth, pos_category, target_word, count=4):
    """Formats a subset of verified entries structurally matched to the target word."""
    examples = ""
    matched_entries = []

    # 1. Determine the structural template to hunt for based on the word's ending
    structural_target = ""
    if pos_category == "Verb":
        if target_word.endswith("ಿಸು"):
            structural_target = "kn-conj-isu"
        elif target_word.endswith("ಕೊಳ್ಳು"):
            structural_target = "kn-conj-koḷḷu"
        elif target_word[-1] in ["ಿ", "ೆ", "ೈ"]:
            # Target for verbs like ಕುಡಿ (kuḍi) or ಬರೆ (bare)
            structural_target = "kn-conj-e-i-other"
        else:
            structural_target = "kn-conj-u"  # General catch for native -u verbs

    elif pos_category == "Noun":
        last_char = target_word[-1]
        if last_char == "ು":
            structural_target = "kn-decl-u"
        elif last_char in ["ಿ", "ೆ", "ೈ"]:
            structural_target = "kn-decl-e-i-ai"
        else:
            structural_target = "kn-decl-a"

    # 2. PRIORITY PASS: Match POS *and* Structural Template
    for word, wikitext in current_ground_truth.items():
        if f"==={pos_category}===" in wikitext and structural_target in wikitext:
            matched_entries.append((word, wikitext))
        if len(matched_entries) == count:
            break

    # 3. SECONDARY PASS: Fallback to just matching POS (if we don't have enough perfect matches)
    if len(matched_entries) < count:
        for word, wikitext in current_ground_truth.items():
            if f"==={pos_category}===" in wikitext and (word, wikitext) not in matched_entries:
                matched_entries.append((word, wikitext))
            if len(matched_entries) == count:
                break

    # 4. FINAL FALLBACK: Fill with anything available to prevent prompt crashes
    if len(matched_entries) < count:
        for word, wikitext in current_ground_truth.items():
            if (word, wikitext) not in matched_entries:
                matched_entries.append((word, wikitext))
            if len(matched_entries) == count:
                break

    # 5. Format the selected entries
    for i, (word, wikitext) in enumerate(matched_entries):
        examples += f"\nExample {i + 1}:\nWord: {word}\nOutput:\n{wikitext}\n---\n"

    return examples


def get_template_logic(word, pos):
    last_char = word[-1]

    if pos == "Noun":
        if last_char == "ು":  # Ends in -u
            # 'manju' -> 'manja' happens automatically in the template
            stem = word[:-1]
            return f"{{{{kn-decl-u|{word}|{stem}}}}}"
        elif last_char in ["ಿ", "ೆ", "ೈ"]:  # Ends in -i, -e, -ai
            return f"{{{{kn-decl-e-i-ai|{word}|{word}}}}}"
        else:
            # Default for -a stems (includes inherent 'a' in consonants like ನ, ಮ, ಲ)
            return f"{{{{kn-decl-a|{word}}}}}"

    elif pos == "Verb":
        # 1. Reflexive compound verbs (e.g., ನೋಡಿಕೊಳ್ಳು)
        if word.endswith("ಕೊಳ್ಳು"):
            prefix = word[:-5]  # Drops the 5 characters of 'ಕೊಳ್ಳು'
            return f"{{{{kn-conj-koḷḷu|{prefix}}}}}"

        # 2. Causative or loanword verbs ending in -isu (e.g., ಕಳುಹಿಸು, ಅನಿಸು)
        if word.endswith("ಿಸು"):
            stem = word[:-1]
            return f"{{{{kn-conj-isu|{word}|{stem}}}}}"

        # 3. Verbs ending in -i, -e, or -ai (e.g., ಕುಡಿ, ಬರೆ)
        if last_char in ["ಿ", "ೆ", "ೈ"]:
            # Pattern: word | present_stem (+ya) | past_stem (+da) | imperative
            return f"{{{{kn-conj-e-i-other|{word}|{word}ಯ|{word}ದ|{word}}}}}"

        # 4. Standard native -u verbs (e.g., ಮಾಡು, ನಗು, ಕೊಡು)
        # We tell the AI to look for the double consonant in the past stem.
        return "IRREGULAR_CHECK"

    return "Template not found"


# --- SYSTEM PROMPT ---
SYSTEM_PROMPT = """
You are a Kannada Lexicographer specializing in Wiktionary formatting.

I. FORMATTING RULES:
Use ONLY Kannada script for Kannada words. NEVER use Devanagari for Kannada words.
Use ONLY Devanagari for Sanskrit words.
For Nouns ending in -u, use {{kn-decl-u|FullWord|Stem}} (e.g., {{kn-decl-u|ಊರು|ಊರ}}).
For Nouns ending in -i, -e, or -ai, use {{kn-decl-e-i-ai|FullWord|FullWord}}.

II. VERB TEMPLATE LOGIC:
If a native verb is IRREGULAR (past participle ends in a double consonant like ನಕ್ಕು, ಕೊಟ್ಟು, ಬಿಟ್ಟು), use {{kn-conj-u-irreg|PresentStem|PastParticiple|PastStem}}.
If a verb ends in -ಿಸು, use {{kn-conj-isu|FullWord|Stem}} (e.g., {{kn-conj-isu|ಓಡಿಸು|ಓಡಿಸ}}).
For all other regular verbs ending in -u, use {{kn-conj-u|PresentStem|PastParticiple|PastStem}}.

III. ETYMOLOGY RULES:
If a word starts with ಪ್ರ- (pra-), ವಿ- (vi-), or ಸಂ- (sam-), it is likely a Sanskrit loan. Use {{bor|kn|sa|SanskritWord}}.
Do NOT invent Proto-Dravidian roots. If unsure, cite the components of the compound using {{compound|kn|Part1|Part2}}.

IV. GENDER RULES:
Inanimate objects and abstract concepts are ALWAYS neuter (g=n).

V. EXAMPLE SENTENCE RULES:
1. Keep sentences SHORT and SIMPLE (maximum 8 words).
2. Use the Subject-Object-Verb (SOV) structure.
3. Avoid passive voice. Use common everyday contexts.

VI. CONSTRAINTS:
Output raw Wikitext ONLY. No introductory remarks.
"""

# --- APP UI ---
st.set_page_config(page_title="Kannada Wiktionary Gen", page_icon="🌿")
st.title("Kannada Wiktionary Generator")

word = st.text_input("Enter a Kannada word:")
translation = st.text_input("Enter the English translation:")

# Clear previous result & sandbox if a new word is typed
if "last_word" not in st.session_state or st.session_state["last_word"] != word:
    st.session_state["last_word"] = word
    if 'current_result' in st.session_state:
        del st.session_state['current_result']
    if 'sandbox_results' in st.session_state:
        del st.session_state['sandbox_results']

pos_category = st.selectbox("Select Part of Speech:", ["Noun", "Verb", "Adjective", "Adverb"])

if word:
    ground_truth = load_ground_truth()

    if word in ground_truth:
        st.success("Found in Ground Truth!")
        st.text_area("Wiktionary Entry:", ground_truth[word], height=400)
    else:
        if st.button("Generate Wikitext"):
            with st.spinner(f"Analyzing '{word}'..."):
                try:
                    template_instruction = get_template_logic(word, pos_category)

                    if template_instruction == "IRREGULAR_CHECK":
                        template_instruction = "If the past participle has a double consonant (geminate), use {{kn-conj-u-irreg|Stem|Participle|PastStem}}. Otherwise use {{kn-conj-u}}."

                    examples_block = get_few_shot_examples(ground_truth, pos_category, word)

                    full_prompt = (
                        f"Use these examples as a formatting guide:\n{examples_block}\n\n"
                        f"CRITICAL CONSTRAINTS:\n"
                        f"- The word is '{word}'.\n"
                        f"- Its English translation is '{translation}'.\n"
                        f"- TEMPLATE DIRECTIVE: {template_instruction}\n\n"
                        f"Now, generate a Wiktionary entry for the word: {word}"
                    )

                    response = ollama.chat(model='qwen2.5:14b', messages=[
                        {'role': 'system', 'content': SYSTEM_PROMPT},
                        {'role': 'user', 'content': full_prompt}
                    ])

                    if response.get('message', {}).get('content'):
                        st.session_state['current_result'] = response['message']['content']
                    else:
                        st.error("AI returned an empty response. Please try again.")

                except Exception as e:
                    st.error(f"Error: {e}")

    if 'current_result' in st.session_state:
        st.subheader("Edit & Verify")
        edited_entry = st.text_area("Final Wikitext:", st.session_state['current_result'], height=400)

        # --- SENTENCE SANDBOX ---
        if st.button("Generate 3 Example Sentences"):
            with st.spinner("Brainstorming simple sentences..."):
                sandbox_prompt = (
                    f"Word: {word}\n"
                    f"Meaning: {translation}\n\n"
                    "Task: Write 3 short, simple, everyday Kannada sentences using the word accurately.\n"
                    f"CRITICAL RULE: The exact Kannada word '{word}' or its conjugated/declined forms MUST be used in every single sentence.\n"
                    "Formatting: Use {{ux|kn|Kannada|t=English}}\n\n"
                    "Examples:\n"
                    "1. {{ux|kn|ರೈಲು ನಿಲ್ದಾಣ ಎಲ್ಲಿದೆ?|t=Where is the train station?}}\n"
                    "2. {{ux|kn|ನಾನು ರೈಲಿನಲ್ಲಿ ಪ್ರಯಾಣಿಸುತ್ತೇನೆ.|t=I travel by train.}}\n\n"
                    "Rules:\n"
                    "- Maximum 8 words per sentence.\n"
                    "- Strictly use Subject-Object-Verb order.\n"
                    "- OUTPUT ONLY THE 3 FORMATTED WIKITEXT LINES.\n"
                    "- EACH sentence MUST be on a SEPARATE LINE.\n"  # <--- NEW RULE
                    "- NO introductory text, NO explanations, NO conversational filler.\n"
                    "- STRICTLY restrict output to Kannada and English."
                )
                try:
                    resp = ollama.chat(model='qwen2.5:14b', messages=[
                        {'role': 'system',
                         'content': 'You are a strict Kannada linguistic assistant. Output raw Wikitext ONLY. Ensure sentences are natural, grammatically correct, and used in everyday speech. Never invent words.'},
                        {'role': 'user', 'content': sandbox_prompt}
                    ])
                    st.session_state['sandbox_results'] = resp['message']['content']
                except Exception as e:
                    st.error(f"Error generating sentences: {e}")

        if 'sandbox_results' in st.session_state:
            # This renders the output exactly as received, preserving newlines
            st.code(st.session_state['sandbox_results'], language='text')

        # --- SAVE BUTTON ---
        st.markdown("---")
        if st.button("Save to Ground Truth"):
            save_to_ground_truth(word, edited_entry)
            st.balloons()
            st.success(f"Verified entry for '{word}' added to your knowledge base!")
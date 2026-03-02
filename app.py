import streamlit as st
import json
import ollama
import os
import time
import re
import requests
from bs4 import BeautifulSoup
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
        'ಱ': 'ṟ', 'ೞ': 'ḻ'
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
            if i + 1 < len(text):
                next_char = text[i + 1]
                if next_char not in KANNADA_MATRAS and next_char != VIRAMA:
                    result += 'a'
            else:
                result += 'a'
        elif char in KANNADA_VOWELS:
            result += KANNADA_VOWELS[char]
        elif char in KANNADA_MATRAS:
            result += KANNADA_MATRAS[char]
        elif char == VIRAMA:
            pass
        elif char in KANNADA_MODIFIERS:
            result += KANNADA_MODIFIERS[char]
        elif char in [ZWNJ, ZWJ]:
            pass
        else:
            result += char

    return result

def transliterate_cognate(word, lang_code):
    """Maps Kannada script to native Indic scripts for Wiktionary {{cog}} templates."""
    if lang_code == 'te':
        # Telugu (Unicode offset: subtract 0x0080 from Kannada)
        return "".join(chr(ord(c) - 0x0080) if 0x0C80 <= ord(c) <= 0x0CFF else c for c in word)

    elif lang_code == 'ml':
        # Malayalam (Unicode offset: add 0x0080 to Kannada)
        return "".join(chr(ord(c) + 0x0080) if 0x0C80 <= ord(c) <= 0x0CFF else c for c in word)

    elif lang_code == 'mr':
        # Marathi / Devanagari (Unicode offset: subtract 0x0380 from Kannada)
        return "".join(chr(ord(c) - 0x0380) if 0x0C80 <= ord(c) <= 0x0CFF else c for c in word)

    elif lang_code == 'ta':
        # Tamil (Requires manual mapping due to missing aspirated/voiced consonants)
        ta_map = {
            'ಅ': 'அ', 'ಆ': 'ஆ', 'ಇ': 'இ', 'ಈ': 'ஈ', 'ಉ': 'உ', 'ಊ': 'ஊ', 'ಋ': 'ரு',
            'ಎ': 'எ', 'ಏ': 'ஏ', 'ಐ': 'ஐ', 'ಒ': 'ஒ', 'ಓ': 'ஓ', 'ಔ': 'ஔ',
            'ಕ': 'க', 'ಖ': 'க', 'ಗ': 'க', 'ಘ': 'க', 'ಙ': 'ங',
            'ಚ': 'ச', 'ಛ': 'ச', 'ಜ': 'ஜ', 'ಝ': 'ஜ', 'ಞ': 'ஞ',
            'ಟ': 'ட', 'ಠ': 'ட', 'ಡ': 'ட', 'ಢ': 'ட', 'ಣ': 'ண',
            'ತ': 'த', 'ಥ': 'த', 'ದ': 'த', 'ಧ': 'த', 'ನ': 'ந',
            'ಪ': 'ப', 'ಫ': 'ப', 'ಬ': 'ப', 'ಭ': 'ப', 'ಮ': 'ம',
            'ಯ': 'ய', 'ರ': 'ர', 'ಲ': 'ல', 'ವ': 'வ',
            'ಶ': 'ஶ', 'ಷ': 'ஷ', 'ಸ': 'ஸ', 'ಹ': 'ஹ',
            'ಳ': 'ள', 'ಱ': 'ற', 'ೞ': 'ழ',
            'ಾ': 'ா', 'ಿ': 'ி', 'ೀ': 'ீ', 'ು': 'ு', 'ೂ': 'ூ', 'ೃ': '்ரு',
            'ೆ': 'ெ', 'ೇ': 'ே', 'ೈ': 'ை', 'ೊ': 'ொ', 'ೋ': 'ோ', 'ಔ': 'ௌ',
            '್': '்', 'ಂ': 'ம்', 'ಃ': 'ஃ'
        }
        return "".join(ta_map.get(c, c) for c in word)

    else:
        # Tulu (tcy) is traditionally written in Kannada script on Wiktionary
        return word

def source_etymology(target_word):
    """
    Scrapes DSAL Kittel dictionary by targeting specific divs and filtering for exact headwords.
    Extracts explicit etymology markers, identifies Dravidian cognates (with strict formatting checks),
    and falls back to an orthographical check for Sanskrit characters.
    Returns: (etymology_template_string, cognate_text_string, debug_logs_list)
    """
    fallback_template = "{{rfe|kn}}"
    word = target_word.strip()
    debug_logs = []
    cognate_text = ""

    try:
        url = "https://dsal.uchicago.edu/cgi-bin/app/kittel_query.py"
        params = {'qs': word, 'searchhws': 'yes'}
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}

        req = requests.Request('GET', url, params=params, headers=headers).prepare()
        debug_logs.append(f"{req.url}")

        response = requests.Session().send(req, timeout=10)

        if response.status_code != 200:
            debug_logs.append("Failed to fetch page. Non-200 status code.")
            return fallback_template, cognate_text, debug_logs

        soup = BeautifulSoup(response.text, 'html.parser')

        # --- 1. GLOBAL "NO RESULTS" CHECK ---
        query_display = soup.find(id='query_display')
        query_text = query_display.get_text().lower() if query_display else soup.get_text().lower()

        if "no results for search term" in query_text:
            debug_logs.append("Database confirmed 0 results for this word.")
            return fallback_template, cognate_text, debug_logs

        # --- 2. ISOLATION LOGIC (STRICT HEADWORD MATCHING) ---
        results_block = soup.find(id='results_display')

        if not results_block:
            debug_logs.append("No <div id='results_display'> found. Word likely has no entries.")
            return fallback_template, cognate_text, debug_logs

        valid_text_blocks = []
        for hw_div in results_block.find_all('div', class_='hw_result'):
            headword_link = hw_div.find('a')
            if headword_link and headword_link.get_text(strip=True) == word:
                valid_text_blocks.append(hw_div.get_text().lower())

        if not valid_text_blocks:
            debug_logs.append(
                f"Match Triggered: Results found, but no EXACT headword match for '{word}'. Assumed Native.")
            return "{{inh|kn|dra-pro|}}", cognate_text, debug_logs

        results_text = " ".join(valid_text_blocks)
        debug_logs.append(f"Isolated {len(valid_text_blocks)} exact match entries. Analyzing plain text...")

        # --- 3. COGNATE EXTRACTION (CLUSTER MATCHING) ---
        # FIX 1: Added \b (word boundary) to ensure it doesn't grab the end of English words like 'imminent.'
        # FIX 2: Removed a-z from the second capture group. It now strictly requires a Kannada-script word.
        cluster_pattern = r'((?:(?:\bm|\bt|\bt[eĕě]|\btu|\bmhr)\.\s*,?\s*)+)([\u0C80-\u0CFF\-]+)'

        lang_map = {
            'm.': ('Malayalam', 'ml'),
            't.': ('Tamil', 'ta'),
            'te.': ('Telugu', 'te'),
            'tĕ.': ('Telugu', 'te'),
            'tě.': ('Telugu', 'te'),
            'tu.': ('Tulu', 'tcy'),
            'mhr.': ('Marathi', 'mr')
        }

        cognates_found = []
        seen_cognates = set()

        if re.search(r'\(\s*c\.\s*[;,]', results_text):
            debug_logs.append("Marker Found: '(c.;' - Word is marked as common alongside sister languages.")

        for match in re.finditer(cluster_pattern, results_text):
            abbrev_cluster = match.group(1).lower()
            cog_word = match.group(2).strip()

            # Using \b here as well to cleanly isolate the abbreviations
            individual_abbrevs = re.findall(r'(?:\bm|\bt|\bt[eĕě]|\btu|\bmhr)\.', abbrev_cluster)

            for abbr in individual_abbrevs:
                if abbr in lang_map:
                    lang_name, lang_code = lang_map[abbr]
                    native_script_word = transliterate_cognate(cog_word, lang_code)

                    cog_str = f"{{{{cog|{lang_code}|{native_script_word}}}}}"

                    if cog_str not in seen_cognates:
                        seen_cognates.add(cog_str)
                        cognates_found.append(cog_str)

        if cognates_found:
            cognate_text = "Cognate with " + ", ".join(cognates_found) + "."
            debug_logs.append(f"Cognates successfully built: {cognate_text}")
        else:
            debug_logs.append("No sister-language cognates identified in the text.")

        # --- 4. ETYMOLOGY REGEX MARKERS ---
        def match_marker(patterns):
            for pattern in patterns:
                match = re.search(pattern, results_text)
                if match:
                    return match.group(0)
            return None

        # Cleaned out English and Portuguese
        tadbhava_patterns = [r'\btbh\.', r'\btadbhava\b']
        tatsama_patterns = [r'\bts\.', r'\bsk\.', r'\btatsama\b']
        hindustani_patterns = [r'\bh\.', r'\bhindustani\b']

        matched_tbh = match_marker(tadbhava_patterns)
        matched_ts = match_marker(tatsama_patterns)
        matched_h = match_marker(hindustani_patterns)

        if matched_tbh:
            debug_logs.append(f"Match Triggered: Tadbhava (Sanskrit Borrowing). Found tag: '{matched_tbh}'")
            return "{{bor|kn|sa|}}", cognate_text, debug_logs
        elif matched_ts:
            debug_logs.append(f"Match Triggered: Tatsama (Direct Sanskrit Borrowing). Found tag: '{matched_ts}'")
            return "{{bor|kn|sa|}}", cognate_text, debug_logs
        elif matched_h:
            debug_logs.append(f"Match Triggered: Hindustani/Persian Loan. Found tag: '{matched_h}'")
            return "{{bor|kn|hi|}}", cognate_text, debug_logs

        # --- 5. ORTHOGRAPHY HEURISTIC (SANSKRIT FALLBACK) ---
        sanskrit_chars = ['ಖ', 'ಘ', 'ಛ', 'ಝ', 'ಠ', 'ಢ', 'ಥ', 'ಧ', 'ಫ', 'ಭ', 'ಶ', 'ಷ', 'ಋ', 'ಃ']
        if any(char in word for char in sanskrit_chars):
            debug_logs.append(
                "Match Triggered: No explicit markers, but word contains Sanskrit characters. Assumed Tatsama/loan.")
            return "{{bor|kn|sa|}}", cognate_text, debug_logs

        debug_logs.append(
            "Match Triggered: Word found, no foreign markers, pure Dravidian orthography. Assumed Native Dravidian.")
        return "{{inh|kn|dra-pro|}}", cognate_text, debug_logs

    except Exception as e:
        debug_logs.append(f"Exception: {str(e)}")
        return fallback_template, cognate_text, debug_logs


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

# Word Details
word = st.text_input("Enter word:")
translation = st.text_input("Enter translation:")
pos_categories = st.multiselect("POS:", ["Noun", "Verb", "Adjective", "Adverb"], default=["Noun"])
st.markdown("---")

# Clear session state if a new word is typed so old etymologies don't bleed over
if word:
    if 'current_search_word' not in st.session_state or st.session_state['current_search_word'] != word:
        st.session_state['current_search_word'] = word
        st.session_state.pop('etymology_line', None)
        st.session_state.pop('etymology_logs', None)
        st.session_state.pop('cognate_text', None)
        st.session_state.pop('current_result', None)

    ground_truth = load_ground_truth()
    if word in ground_truth:
        st.success("Found in Ground Truth!")
        st.text_area("Entry:", ground_truth[word], height=400)
    else:
        # --- PIPELINE STEP 1: ETYMOLOGY ---
        st.markdown("### Step 1: Etymology Lookup")
        if st.button("🔍 Fetch Etymology"):
            with st.spinner("Querying University of Chicago DSAL..."):
                etymology_line, cognate_text, debug_logs = source_etymology(word)

                st.session_state['etymology_line'] = etymology_line
                st.session_state['cognate_text'] = cognate_text
                st.session_state['etymology_logs'] = debug_logs

        # If etymology has been fetched, show it
        if 'etymology_line' in st.session_state:
            with st.expander("🕰️ Etymology Extraction Results", expanded=True):
                st.markdown(f"**Extracted Template:** `{st.session_state['etymology_line']}`")

                if st.session_state['cognate_text']:
                    st.markdown(f"**Identified Cognates:** {st.session_state['cognate_text']}")

                st.markdown("#### 📝 Etymology Search Notes")
                for log in st.session_state['etymology_logs']:
                    st.code(log, language="text")

            # --- PIPELINE STEP 2: GENERATION ---
            st.divider()
            st.markdown("### Step 2: Entry Assembly")

            # --- MOVED: User-Provided Example Sentence ---
            st.markdown("#### 📝 Manual Example Sentence (Optional)")
            st.caption("Provide your own example sentence here to completely bypass the AI generation step.")
            user_kn = st.text_input("Kannada Sentence:")
            user_en = st.text_input("English Translation:")
            st.markdown("---")

            if st.button("✍️ Generate Wiktionary Entry"):
                st.markdown("### 🔍 Assembly Audit Trail")
                try:
                    start_time = time.time()

                    etymology_line = st.session_state['etymology_line']
                    cognate_line = st.session_state['cognate_text']

                    # --- PYTHON PRE-COMPUTATION ---
                    morphology_block = get_dynamic_morphology_block(word, pos_categories)
                    if "IRREGULAR_CHECK" in morphology_block:
                        stem = word.removesuffix("ು")
                        morphology_block = morphology_block.replace("IRREGULAR_CHECK",
                                                                    f"{{{{kn-conj-u|{word}|{stem}ಿ|{stem}ಿದ}}}}")

                    primary_pos = pos_categories[0] if pos_categories else "Noun"
                    pos_template = f"{{{{kn-{primary_pos.lower()}}}}}"
                    formatted_translation = ", ".join([f"[[{t.strip()}]]" for t in translation.split(",") if t.strip()])

                    with st.expander("⚙️ Python Deterministic Logic", expanded=False):
                        st.write(f"**Morphology Template:** `{morphology_block}`")
                        st.write(f"**Definitions:** `{formatted_translation}`")

                    # --- CONDITIONAL AI GENERATION ---
                    if user_kn and user_en:
                        st.success("💡 User-provided sentence detected. Bypassing AI generation.")
                        final_kn = user_kn.strip()
                        final_en = user_en.strip()
                    else:
                        st.info("🤖 No manual sentence provided. Generating via Ollama...")
                        examples_block = get_few_shot_sentences(ground_truth, word)

                        # --- STEP 1: DRAFTING (27B) ---
                        with st.expander("📝 Step 1: Drafter (27B)", expanded=True):
                            step1_placeholder = st.empty()
                            drafter_user_content = f"### Context Examples:\n{examples_block}\n\n### Target Word: {word}\nMeaning: {translation}\n"
                            draft_output = ""
                            try:
                                for chunk in ollama.chat(model=DRAFTER_MODEL,
                                                         messages=[{'role': 'system', 'content': DRAFTER_PROMPT},
                                                                   {'role': 'user', 'content': drafter_user_content}],
                                                         stream=True):
                                    draft_output += chunk['message']['content']
                                    step1_placeholder.markdown(draft_output)
                            except Exception as e:
                                st.error(
                                    f"Ollama Connection Error: {e}. The local model crashed. Please restart Ollama or type a manual sentence above.")
                                st.stop()

                            kn_draft, en_draft = parse_kannada_english(draft_output)

                        # --- STEP 2: LOGIC AUDIT (27B) ---
                        with st.expander("🛡️ Step 2: Logic Auditor (27B)", expanded=True):
                            step2_placeholder = st.empty()
                            logic_user_content = f"Target Word: {word}\nRequired Meaning: {translation}\n\nDraft:\nKANNADA: {kn_draft}\nENGLISH: {en_draft}"
                            logic_output = ""
                            try:
                                for chunk in ollama.chat(model=LOGIC_MODEL,
                                                         messages=[{'role': 'system', 'content': LOGIC_AUDITOR_PROMPT},
                                                                   {'role': 'user', 'content': logic_user_content}],
                                                         stream=True):
                                    logic_output += chunk['message']['content']
                                    step2_placeholder.markdown(logic_output)
                            except Exception as e:
                                st.error(f"Ollama Connection Error: {e}.")
                                st.stop()

                            final_kn, final_en = parse_kannada_english(logic_output)
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
{cognate_line}

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
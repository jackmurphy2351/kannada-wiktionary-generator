import streamlit as st
import time
import ollama
from dotenv import load_dotenv

from data_manager import load_ground_truth, save_to_ground_truth, format_time
from linguistics import get_dynamic_morphology_block, transliterate_kannada_to_iso
from scraper import source_etymology
from wiktionary_api import check_wiktionary_entry_exists, upload_to_wiktionary
from llm_service import DRAFTER_PROMPT, LOGIC_AUDITOR_PROMPT, get_few_shot_sentences, parse_kannada_english

load_dotenv()

st.set_page_config(page_title="Kannada Wiktionary Gen", page_icon="🌿")

DRAFTER_MODEL = 'translategemma:27b'
LOGIC_MODEL = 'translategemma:27b'

st.title("Kannada Wiktionary Generator")

word = st.text_input("Enter word:")
translation = st.text_input("Enter translation:")
pos_categories = st.multiselect("POS:", ["Noun", "Verb", "Adjective", "Adverb"], default=["Noun"])
st.markdown("---")

if word:
    with st.spinner("Checking Wiktionary database..."):
        exists = check_wiktionary_entry_exists(word)
        if exists is True:
            wiktionary_url = f"https://en.wiktionary.org/wiki/{word}"
            st.warning(f"🔴 An entry for **{word}** already exists on Wiktionary! [View the existing entry here]({wiktionary_url}). Proceed with caution.")
        elif exists is False:
            st.success(f"🟢 No entry found for **{word}**. Safe to create a new page!")

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
        st.markdown("### Step 1: Etymology Lookup")
        if st.button("🔍 Fetch Etymology"):
            with st.spinner("Querying University of Chicago DSAL..."):
                etymology_line, cognate_text, debug_logs = source_etymology(word)
                st.session_state['etymology_line'] = etymology_line
                st.session_state['cognate_text'] = cognate_text
                st.session_state['etymology_logs'] = debug_logs

        if 'etymology_line' in st.session_state:
            with st.expander("🕰️ Etymology Extraction Results", expanded=True):
                st.markdown(f"**Extracted Template:** `{st.session_state['etymology_line']}`")
                if st.session_state['cognate_text']:
                    st.markdown(f"**Identified Cognates:** {st.session_state['cognate_text']}")
                st.markdown("#### 📝 Etymology Search Notes")
                for log in st.session_state['etymology_logs']:
                    st.code(log, language="text")

            st.divider()
            st.markdown("### Step 2: Entry Assembly")
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

                    morphology_block = get_dynamic_morphology_block(word, pos_categories)
                    if "IRREGULAR_CHECK" in morphology_block:
                        stem = word.removesuffix("ು")
                        morphology_block = morphology_block.replace("IRREGULAR_CHECK", f"{{{{kn-conj-u|{word}|{stem}ಿ|{stem}ಿದ}}}}")

                    primary_pos = pos_categories[0] if pos_categories else "Noun"
                    pos_template = f"{{{{kn-{primary_pos.lower()}}}}}"
                    formatted_translation = ", ".join([f"[[{t.strip()}]]" for t in translation.split(",") if t.strip()])

                    with st.expander("⚙️ Python Deterministic Logic", expanded=False):
                        st.write(f"**Morphology Template:** `{morphology_block}`")
                        st.write(f"**Definitions:** `{formatted_translation}`")

                    if user_kn and user_en:
                        st.success("💡 User-provided sentence detected. Bypassing AI generation.")
                        final_kn = user_kn.strip()
                        final_en = user_en.strip()
                    else:
                        st.info("🤖 No manual sentence provided. Generating via Ollama...")
                        examples_block = get_few_shot_sentences(ground_truth, word)

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
                                st.error(f"Ollama Connection Error: {e}.")
                                st.stop()
                            kn_draft, en_draft = parse_kannada_english(draft_output)

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

                    with st.expander("🔤 Step 3: Transliteration Proofreader (Python Logic)", expanded=True):
                        final_tr = transliterate_kannada_to_iso(final_kn)
                        st.markdown(f"**Target Sentence:** `{final_kn}`")
                        st.markdown(f"**ISO 15919 Transliteration:** `{final_tr}`")

                    st.success(f"Generation Complete! ({format_time(time.time() - start_time)})")

                    # --- ETYMOLOGY FORMATTING FIX ---
                    # Construct the etymology section dynamically to avoid blank lines and weird punctuation
                    formatted_etymology = ""
                    if "{{bor" in etymology_line:
                        formatted_etymology = f"Borrowed from {etymology_line}."
                    elif "{{inh" in etymology_line:
                        formatted_etymology = f"From {etymology_line}."
                    else:
                        formatted_etymology = etymology_line  # Usually just {{rfe|kn}}

                    # Add cognates on the same line if they exist (scraper.py already includes the period)
                    if cognate_line:
                        formatted_etymology += f" {cognate_line}"

                    final_wikitext = f"""==Kannada==

===Etymology===
{formatted_etymology}

===Pronunciation===
* {{{{kn-IPA|{word}}}}}

==={primary_pos}===
{pos_template}
# {formatted_translation}

====Usage notes====
* {{{{ux|kn|{final_kn}|tr={final_tr}|t={final_en}}}}}

{morphology_block}

===References===
* {{{{R:kn:Alar}}}}
* {{{{R:kn:Kittel}}}}"""
                    st.session_state['current_result'] = final_wikitext

                except Exception as e:
                    st.error(f"Pipeline Error: {e}")

    if 'current_result' in st.session_state:
        st.divider()
        st.markdown("### ✨ Final Wiktionary Entry")
        edited_entry = st.text_area("Verify & Edit:", st.session_state['current_result'], height=400)

        col1, col2 = st.columns(2)
        with col1:
            if st.button("💾 Save to Database", use_container_width=True):
                save_to_ground_truth(word, edited_entry)
                st.success("Entry saved successfully!")
        with col2:
            if st.button("🚀 Publish directly to Wiktionary", use_container_width=True):
                with st.spinner("Authenticating and publishing..."):
                    success, message = upload_to_wiktionary(word, edited_entry)
                    if success:
                        st.success(message)
                        st.balloons()
                    else:
                        st.error(message)
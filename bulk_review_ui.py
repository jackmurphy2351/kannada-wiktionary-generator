import streamlit as st
import json
import time
import re
from core.wiktionary_api import upload_to_wiktionary
from core.llm_service import DRAFTER_PROMPT, parse_kannada_english, stream_chat
from core.linguistics import transliterate_kannada_to_iso
from core.data_manager import save_to_ground_truth

STAGING_FILE = 'data/staged_entries.json'
DISCARDED_FILE = 'data/discarded_entries.json' # <-- NEW CONSTANT

st.set_page_config(page_title="Batch Review Queue", page_icon="📝")
st.title("Bulk Entry Review Queue")


def load_staged_data():
    try:
        with open(STAGING_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        return {}


def save_staged_data(data):
    with open(STAGING_FILE, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

def load_discarded_data():
    try:
        with open(DISCARDED_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        return {}

def save_discarded_data(data):
    with open(DISCARDED_FILE, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

def apply_bolding(word, translation, kn_sent, en_sent, tr_sent):
    """Applies Wiktionary bolding (''') to the target word and its translation."""
    bold_marker = "'''"

    # Bold Kannada word
    if word in kn_sent:
        kn_sent = kn_sent.replace(word, f"{bold_marker}{word}{bold_marker}")

    # Bold Transliteration
    word_tr = transliterate_kannada_to_iso(word)
    if word_tr in tr_sent:
        tr_sent = tr_sent.replace(word_tr, f"{bold_marker}{word_tr}{bold_marker}")

    # Bold English translation (first word/phrase)
    primary_en = translation.split(",")[0].strip()
    if primary_en:
        en_sent = re.sub(
            f"(?i)\\b({re.escape(primary_en)})\\b",
            f"{bold_marker}\\1{bold_marker}",
            en_sent
        )

    return kn_sent, en_sent, tr_sent


staged_data = load_staged_data()
pending_words = [word for word, data in staged_data.items() if data["status"] == "pending_review"]

if not pending_words:
    st.success("The queue is empty! Run `bulk_processor.py` to fetch a new batch.")
    st.stop()

# Sidebar for navigation
st.sidebar.header(f"Pending Queue ({len(pending_words)})")
selected_word = st.sidebar.radio("Select a word to review:", pending_words)

entry_data = staged_data[selected_word]
definition = entry_data['definition']

st.markdown(f"### Reviewing: **{selected_word}** ({entry_data['pos']})")
st.caption(f"Intended meaning: {definition}")

col1, col2 = st.columns(2)
with col1:
    if st.button("🤖 Generate Example"):
        with st.spinner("Drafting sentence..."):
            prompt = f"### Target Word: {selected_word}\nMeaning: {definition}\n"
            try:
                draft_output = "".join(stream_chat(DRAFTER_PROMPT, prompt))
            except Exception as e:
                st.error(f"LLM Generation Error: {e}")
                st.stop()
            kn_sent, en_sent = parse_kannada_english(draft_output)
            tr_sent = transliterate_kannada_to_iso(kn_sent)

            # Apply bolding
            kn_sent, en_sent, tr_sent = apply_bolding(selected_word, definition, kn_sent, en_sent, tr_sent)

            # Look specifically for the definition line to insert beneath
            usage_line = f"#: {{{{ux|kn|{kn_sent}|tr={tr_sent}|t={en_sent}}}}}"

            # Look for the first line that starts with exactly '# ' (the definition line)
            pattern = re.compile(r"^(# .*?)$", re.MULTILINE)

            if pattern.search(entry_data["wikitext"]):
                # \1 keeps the original definition line, \n adds a break, and then we append the usage_line
                entry_data["wikitext"] = pattern.sub(rf"\1\n{usage_line}", entry_data["wikitext"], count=1)
                save_staged_data(staged_data)
                st.rerun()
            else:
                st.error(
                    "Could not find the definition line (starting with '# ') in the Wikitext to insert the example.")

with col2:
    with st.popover("✍️ Add Manual Sentence"):
        man_kn = st.text_input("Kannada:")
        man_en = st.text_input("English:")
        if st.button("Insert"):
            tr_sent = transliterate_kannada_to_iso(man_kn)

            # Apply bolding
            man_kn, man_en, tr_sent = apply_bolding(selected_word, definition, man_kn, man_en, tr_sent)

            usage_line = f"#: {{{{ux|kn|{kn_sent}|tr={tr_sent}|t={en_sent}}}}}"

            # Look for the first line that starts with exactly '# ' (the definition line)
            pattern = re.compile(r"^(# .*?)$", re.MULTILINE)

            if pattern.search(entry_data["wikitext"]):
                # \1 keeps the original definition line, \n adds a break, and then we append the usage_line
                entry_data["wikitext"] = pattern.sub(rf"\1\n{usage_line}", entry_data["wikitext"], count=1)
                save_staged_data(staged_data)
                st.rerun()
            else:
                st.error(
                    "Could not find the definition line (starting with '# ') in the Wikitext to insert the example.")

edited_wikitext = st.text_area("Final Wikitext (Edit as needed):", value=entry_data["wikitext"], height=350)

# Ensure manual edits inside the text area are saved to the dictionary so they don't get lost
if edited_wikitext != entry_data["wikitext"]:
    entry_data["wikitext"] = edited_wikitext
    save_staged_data(staged_data)

c1, c2, c3 = st.columns(3)
with c1:
    if st.button("🚀 Publish to Wiktionary", type="primary"):
        success, msg = upload_to_wiktionary(selected_word, edited_wikitext)
        if success:
            # 1. Save to your verified ground truth database
            save_to_ground_truth(selected_word, edited_wikitext)

            # 2. Remove from the staging queue completely
            staged_data.pop(selected_word)
            save_staged_data(staged_data)

            # 3. Construct the URL and display it
            wiktionary_url = f"https://en.wiktionary.org/wiki/{selected_word}"
            st.success(f"Published and saved to ground truth! [View '{selected_word}' on Wiktionary]({wiktionary_url})")

            # Pausing slightly longer to give you time to click the link
            time.sleep(3)
            st.rerun()
        else:
            st.error(msg)
with c3:
    if st.button("🗑️ Discard/Skip"):
        discarded_data = load_discarded_data()

        # Pop the entry out of staged_data and move it to discarded_data
        discarded_data[selected_word] = staged_data.pop(selected_word)
        discarded_data[selected_word]["status"] = "discarded"

        # Save both files
        save_discarded_data(discarded_data)
        save_staged_data(staged_data)

        st.rerun()
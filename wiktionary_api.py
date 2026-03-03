import os
import re
import requests
import streamlit as st
from dotenv import load_dotenv

load_dotenv()


def check_wiktionary_entry_exists(word):
    contact_email = os.getenv("YOUR_EMAIL")
    url = "https://en.wiktionary.org/w/api.php"
    params = {"action": "query", "titles": word, "format": "json"}
    headers = {
        "User-Agent": f"KannadaWiktionaryGenerator/1.0 (https://github.com/jackmurphy2351/kannada-wiktionary-generator; {contact_email})"
    }

    try:
        response = requests.get(url, params=params, headers=headers, timeout=10)
        response.raise_for_status()
        data = response.json()
        pages = data.get("query", {}).get("pages", {})
        if "-1" in pages:
            return False
        return True
    except requests.exceptions.RequestException as e:
        st.error(f"Wiktionary API Error: {e}")
        return None


def insert_alphabetically(existing_text, new_wikitext, target_lang="Kannada"):
    """
    Parses existing Wikitext to find the correct alphabetical placement for the new language.
    Returns the merged text, or None if the target language already exists.
    """
    if not existing_text.strip():
        return new_wikitext

    # Safety Check: Prevent duplicate language sections on the same page
    if re.search(rf"^==\s*{target_lang}\s*==$", existing_text, re.MULTILINE | re.IGNORECASE):
        return None

    # Regex to find all Level 2 Language Headers (e.g., ==Tulu==, ==English==)
    pattern = re.compile(r"^==\s*([a-zA-Z -]+)\s*==\s*$", re.MULTILINE)
    insert_pos = len(existing_text)

    # Iterate through all language headers on the page
    for match in pattern.finditer(existing_text):
        lang = match.group(1).strip()
        # If we hit a language that comes alphabetically AFTER Kannada, we found our insertion point
        if lang.lower() > target_lang.lower():
            insert_pos = match.start()
            break

    # If Kannada belongs at the very end of the page
    if insert_pos == len(existing_text):
        return existing_text.rstrip() + "\n\n----\n\n" + new_wikitext

    # If Kannada belongs somewhere in the middle (or beginning)
    else:
        before_text = existing_text[:insert_pos].rstrip()
        after_text = existing_text[insert_pos:].lstrip()

        # If there's content before it, sandwich it with horizontal rules
        if before_text:
            return before_text + "\n\n----\n\n" + new_wikitext + "\n\n----\n\n" + after_text
        # If Kannada is the very first language alphabetically
        else:
            return new_wikitext + "\n\n----\n\n" + after_text


def upload_to_wiktionary(word, new_wikitext):
    url = "https://en.wiktionary.org/w/api.php"
    username = os.getenv("WIKTIONARY_USERNAME")
    password = os.getenv("WIKTIONARY_PASSWORD")
    contact_email = os.getenv("YOUR_EMAIL")

    if not username or not password:
        return False, "Wiktionary credentials are missing from the .env file."

    session = requests.Session()
    headers = {
        "User-Agent": f"KannadaWiktionaryGenerator/1.0 (https://github.com/jackmurphy2351/kannada-wiktionary-generator; {contact_email})"
    }
    session.headers.update(headers)

    try:
        # Step 1: Login Token
        res_token = session.get(url, params={"action": "query", "meta": "tokens", "type": "login", "format": "json"})
        res_token.raise_for_status()
        login_token = res_token.json()['query']['tokens']['logintoken']

        # Step 2: Authenticate
        res_login = session.post(url, data={
            "action": "login", "lgname": username, "lgpassword": password,
            "lgtoken": login_token, "format": "json"
        })
        res_login.raise_for_status()
        if res_login.json().get('login', {}).get('result') != 'Success':
            return False, f"Login failed: {res_login.json().get('login', {}).get('reason', 'Unknown error')}"

        # Step 3: CSRF Token
        res_csrf = session.get(url, params={"action": "query", "meta": "tokens", "format": "json"})
        res_csrf.raise_for_status()
        csrf_token = res_csrf.json()['query']['tokens']['csrftoken']

        # Step 4: Fetch existing text
        res_parse = session.get(url, params={"action": "parse", "page": word, "prop": "wikitext", "format": "json"})
        existing_text = ""
        if "error" not in res_parse.json():
            existing_text = res_parse.json()['parse']['wikitext']['*']

        # Step 5: Merge Alphabetically
        final_text = insert_alphabetically(existing_text, new_wikitext, target_lang="Kannada")

        # Catch our duplicate protection trigger
        if final_text is None:
            return False, f"An entry for **Kannada** already exists on the page for '{word}'. Upload aborted to prevent duplication."

        # Step 6: Execute the Edit
        res_edit = session.post(url, data={
            "action": "edit", "title": word, "text": final_text,
            "summary": "Added Kannada entry via KannadaWiktionaryGenerator",
            "token": csrf_token, "bot": True, "format": "json"
        })
        res_edit.raise_for_status()
        edit_data = res_edit.json()

        if "edit" in edit_data and edit_data["edit"].get("result") == "Success":
            return True, f"Successfully published to Wiktionary!"
        else:
            return False, f"Edit failed: {edit_data.get('error', edit_data)}"

    except Exception as e:
        return False, f"API Exception: {str(e)}"
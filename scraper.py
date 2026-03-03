import requests
import re
from bs4 import BeautifulSoup
from linguistics import transliterate_cognate


def source_etymology(target_word):
    """
    Scrapes DSAL Kittel dictionary by targeting specific divs and filtering for exact headwords.
    """
    word = target_word.strip()
    debug_logs = []
    cognate_text = ""

    # --- SANSKRIT ORTHOGRAPHY PRE-CHECK ---
    sanskrit_chars = ['ಖ', 'ಘ', 'ಛ', 'ಝ', 'ಠ', 'ಢ', 'ಥ', 'ಧ', 'ಫ', 'ಭ', 'ಶ', 'ಷ', 'ಋ', 'ಃ']
    has_sanskrit_char = any(char in word for char in sanskrit_chars)

    # Dynamically set the fallback template depending on spelling
    fallback_template = "{{bor|kn|sa|}}" if has_sanskrit_char else "{{rfe|kn}}"

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

        query_display = soup.find(id='query_display')
        query_text = query_display.get_text().lower() if query_display else soup.get_text().lower()

        # --- "NO RESULTS" EXITS ---
        if "no results for search term" in query_text:
            if has_sanskrit_char:
                debug_logs.append(
                    "Database confirmed 0 results. Fallback triggered: Word contains Sanskrit characters. Assumed Tatsama/loan.")
            else:
                debug_logs.append("Database confirmed 0 results for this word.")
            return fallback_template, cognate_text, debug_logs

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

        cluster_pattern = r'((?:(?:\bm|\bt|\bt[eĕě]|\btu|\bmhr)\.\s*,?\s*)+)([\u0C80-\u0CFF\-]+)'
        lang_map = {
            'm.': ('Malayalam', 'ml'), 't.': ('Tamil', 'ta'),
            'te.': ('Telugu', 'te'), 'tĕ.': ('Telugu', 'te'),
            'tě.': ('Telugu', 'te'), 'tu.': ('Tulu', 'tcy'),
            'mhr.': ('Marathi', 'mr')
        }

        cognates_found = []
        seen_cognates = set()

        if re.search(r'\(\s*c\.\s*[;,]', results_text):
            debug_logs.append("Marker Found: '(c.;' - Word is marked as common alongside sister languages.")

        for match in re.finditer(cluster_pattern, results_text):
            abbrev_cluster = match.group(1).lower()
            cog_word = match.group(2).strip()
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

        def match_marker(patterns):
            for pattern in patterns:
                match = re.search(pattern, results_text)
                if match: return match.group(0)
            return None

        matched_tbh = match_marker([r'\btbh\.', r'\btadbhava\b'])
        matched_ts = match_marker([r'\bts\.', r'\bsk\.', r'\btatsama\b'])
        matched_h = match_marker([r'\bh\.', r'\bhindustani\b'])

        if matched_tbh:
            debug_logs.append(f"Match Triggered: Tadbhava (Sanskrit Borrowing). Found tag: '{matched_tbh}'")
            return "{{bor|kn|sa|}}", cognate_text, debug_logs
        elif matched_ts:
            debug_logs.append(f"Match Triggered: Tatsama (Direct Sanskrit Borrowing). Found tag: '{matched_ts}'")
            return "{{bor|kn|sa|}}", cognate_text, debug_logs
        elif matched_h:
            debug_logs.append(f"Match Triggered: Hindustani/Persian Loan. Found tag: '{matched_h}'")
            return "{{bor|kn|hi|}}", cognate_text, debug_logs

        # If no explicit markers were found, we use our pre-calculated heuristic
        if has_sanskrit_char:
            debug_logs.append(
                "Match Triggered: No explicit markers, but word contains Sanskrit characters. Assumed Tatsama/loan.")
            return "{{bor|kn|sa|}}", cognate_text, debug_logs

        debug_logs.append(
            "Match Triggered: Word found, no foreign markers, pure Dravidian orthography. Assumed Native Dravidian.")
        return "{{inh|kn|dra-pro|}}", cognate_text, debug_logs

    except Exception as e:
        debug_logs.append(f"Exception: {str(e)}")
        return fallback_template, cognate_text, debug_logs
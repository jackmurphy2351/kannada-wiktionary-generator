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
        return "".join(chr(ord(c) - 0x0080) if 0x0C80 <= ord(c) <= 0x0CFF else c for c in word)
    elif lang_code == 'ml':
        return "".join(chr(ord(c) + 0x0080) if 0x0C80 <= ord(c) <= 0x0CFF else c for c in word)
    elif lang_code == 'mr':
        return "".join(chr(ord(c) - 0x0380) if 0x0C80 <= ord(c) <= 0x0CFF else c for c in word)
    elif lang_code == 'ta':
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
        return word
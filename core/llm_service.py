import re

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
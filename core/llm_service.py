import re

DRAFTER_PROMPT = """
You are a Kannada linguist.
TASK: Write ONE short Kannada sentence. 

CRITICAL RULE: You MUST include the exact target word in your Kannada sentence.

Output ONLY in this exact format:
KANNADA: [Short sentence containing the target word]
ENGLISH: [English translation]
"""

LOGIC_AUDITOR_PROMPT = """
You are a Kannada grammar editor.
TASK: Fix any grammar or agreement errors in the provided sentence.

CRITICAL RULES:
1. The Kannada sentence MUST contain the exact target word. Add it if it is missing.
2. Only fix genuine errors (like subject-verb or gender agreement).
3. Do not add any conversational filler.

Output ONLY in this exact format:
KANNADA: [Corrected sentence containing the target word]
ENGLISH: [English translation]
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
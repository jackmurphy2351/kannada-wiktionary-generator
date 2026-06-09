import json
import os
import re

import ollama
import requests

DRAFTER_PROMPT = """You are a Kannada lexicographer writing ONE dictionary example.

You are given a Kannada TARGET WORD (already in Kannada script) and its English meaning.
Write a single short, natural Kannada sentence that actually uses the TARGET WORD
(an inflected form of the same word is acceptable), then give its English translation.

STRICT RULES:
- The sentence MUST contain the given Kannada TARGET WORD itself. Never replace it with a
  synonym or a different word, even if the English meaning could be expressed another way.
- Produce EXACTLY ONE example. Do not output a list, multiple sentences, or any commentary.
- Output ONLY these two lines and then STOP:
KANNADA: <one sentence containing the target word>
ENGLISH: <its English translation>
"""

SARVAM_CHAT_URL = "https://api.sarvam.ai/v1/chat/completions"

# Halt generation as soon as the model tries to start a second example. Few-shot
# blocks are labelled "Target Word:" / "---", so the model mirrors those when it runs on.
STOP_SEQUENCES = ["Target Word:", "\n---", "###", "Context Examples"]

# One Kannada sentence + its translation is short; cap output so a runaway model
# cannot generate an endless list (and to bound API cost/latency).
MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "256"))


def stream_chat(system_prompt, user_content):
    """Yield text deltas for a single chat turn from the configured backend.

    Backend is selected by the LLM_BACKEND env var ("sarvam" by default,
    "ollama" for the offline/free fallback). Both UIs iterate over this.
    """
    backend = os.getenv("LLM_BACKEND", "sarvam").lower()
    if backend == "ollama":
        yield from _ollama_stream(system_prompt, user_content)
    else:
        yield from _sarvam_stream(system_prompt, user_content)


def _ollama_stream(system_prompt, user_content):
    model = os.getenv("OLLAMA_DRAFTER_MODEL", "translategemma:27b")
    try:
        for chunk in ollama.chat(
            model=model,
            messages=[{"role": "system", "content": system_prompt},
                      {"role": "user", "content": user_content}],
            options={"stop": STOP_SEQUENCES, "num_predict": MAX_TOKENS},
            stream=True,
        ):
            yield chunk["message"]["content"]
    except Exception as e:
        raise RuntimeError(f"Ollama generation failed (model '{model}'): {e}")


def _sarvam_stream(system_prompt, user_content):
    api_key = os.getenv("SARVAM_API_KEY")
    if not api_key:
        raise RuntimeError(
            "SARVAM_API_KEY is not set. Add it to .env or set LLM_BACKEND=ollama "
            "to use the local fallback."
        )
    model = os.getenv("SARVAM_MODEL", "sarvam-m")
    payload = {
        "model": model,
        "messages": [{"role": "system", "content": system_prompt},
                     {"role": "user", "content": user_content}],
        "max_tokens": MAX_TOKENS,
        "stop": STOP_SEQUENCES,
        # Sarvam reasoning models default to "think" mode, which burns the token
        # budget on hidden reasoning and emits nothing in `content`. We only need a
        # one-sentence answer, so disable reasoning entirely.
        "reasoning_effort": None,
        "stream": True,
    }
    headers = {"Authorization": f"Bearer {api_key}",
               "Content-Type": "application/json"}
    try:
        with requests.post(SARVAM_CHAT_URL, json=payload, headers=headers,
                           stream=True, timeout=120) as resp:
            resp.raise_for_status()
            for raw in resp.iter_lines(decode_unicode=True):
                if not raw or not raw.startswith("data:"):
                    continue
                data = raw[len("data:"):].strip()
                if data == "[DONE]":
                    break
                try:
                    delta = json.loads(data)["choices"][0]["delta"]
                except (json.JSONDecodeError, KeyError, IndexError):
                    continue
                content = delta.get("content")
                if content:
                    yield content
    except requests.RequestException as e:
        raise RuntimeError(f"Sarvam API request failed (model '{model}'): {e}")

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
    """Extract only the FIRST Kannada/English pair, ignoring any trailing examples."""
    kn, en = "", ""
    if "KANNADA:" in text:
        kn = text.split("KANNADA:", 1)[1].split("ENGLISH:", 1)[0]
    if "ENGLISH:" in text:
        en = text.split("ENGLISH:", 1)[1]
        # Cut off anything that begins a second example or a new line block.
        for sep in ("KANNADA:", "Target Word:", "\n"):
            en = en.split(sep, 1)[0]
    clean = lambda s: s.replace("```", "").strip().strip("[]").strip()
    return clean(kn), clean(en)
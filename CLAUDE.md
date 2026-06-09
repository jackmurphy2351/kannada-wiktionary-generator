# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

The **Kannada Wiktionary Generator** is a modular pipeline for creating standardized Wiktionary entries for Kannada words. It combines web scraping (DSAL Kittel Dictionary), deterministic linguistic rules, Unicode manipulation, and optional LLM generation via a pluggable backend (Sarvam hosted API by default, local Ollama as fallback).

The system supports both single-entry and bulk processing workflows with human-in-the-loop review before publishing to Wiktionary via MediaWiki API.

## Architecture

### Core Pipeline Flow

```
Input Word → Etymology Lookup → Morphology Detection → Example Generation → 
Transliteration → Wikitext Assembly → Wiktionary API Upload
```

**Three main execution paths:**
1. **Single-Entry UI** (`app.py`): Draft individual words with full control
2. **Bulk Processor** (`bulk_processor.py`): Headless script that ingests Alar YAML, stages entries
3. **Bulk Review UI** (`bulk_review_ui.py`): Streamlit interface to review and publish staged batches

### Module Structure

| Module | Responsibility |
|--------|-----------------|
| `app.py` | Streamlit single-entry UI. Orchestrates the full pipeline: etymology → morphology → example generation (via Ollama) → transliteration → wikitext assembly. |
| `bulk_processor.py` | Headless batch processor. Loads Alar YAML, groups definitions by word + POS, fetches etymologies, generates MVP wikitext, stages entries to `staged_entries.json`. |
| `bulk_review_ui.py` | Streamlit review queue. Allows manual/AI-generated example sentences, edits wikitext, publishes to Wiktionary, moves discarded entries to `discarded_entries.json`. |
| `core/wiktionary_api.py` | MediaWiki API integration. Checks if entry exists, handles authentication, merges new language sections alphabetically, publishes edits. |
| `core/scraper.py` | DSAL Kittel Dictionary scraper. Extracts etymology templates (`{{bor\|...}}`, `{{inh\|...}}`), identifies cognates in sister languages (Malayalam, Tamil, Telugu, Tulu, Marathi), transliterates them to native scripts. |
| `core/linguistics.py` | Deterministic linguistic rules. Selects morphology templates by analyzing word endings, transliterates Kannada to ISO 15919, maps Kannada script to other Indic scripts for cognates. |
| `core/llm_service.py` | LLM integration. Exposes `stream_chat(system_prompt, user_content)`, which yields text deltas from the backend selected by the `LLM_BACKEND` env var: `sarvam` (hosted, OpenAI-compatible `/v1/chat/completions`) or `ollama` (local fallback). Defines the drafter prompt, parses LLM output into Kannada/English sentence pairs, and provides few-shot examples from ground truth. |
| `core/data_manager.py` | JSON persistence. Loads/saves ground truth (`verified_kannada_entries.json`), formats time durations. |

### Data Files

- `data/alar.yml`: Open-source Kannada dictionary (41MB, must be downloaded separately from Alar Dictionary GitHub)
- `data/verified_kannada_entries.json`: Ground truth database of published entries (used for few-shot examples and deduplication)
- `data/staged_entries.json`: Entries awaiting review (created by bulk_processor, consumed by bulk_review_ui)
- `data/discarded_entries.json`: Entries marked as skip/invalid (prevents reprocessing)

### Key Design Patterns

1. **Etymology Heuristics**: Scraper uses marker matching (`Tbh.`, `Ts.`, `H.`) and orthographic analysis (Sanskrit characters) to classify borrowing origin
2. **Morphology Templates**: Suffix analysis (final vowel character) deterministically selects Wiktionary declension/conjugation template
3. **Script Shifting**: Unicode offset math converts Kannada cognates to Telugu/Tamil/Malayalam/Marathi native scripts
4. **Wiktionary Bolding**: Triple-quote syntax (`'''word'''`) automatically applied to target word, transliteration, and translation in examples
5. **Alphabetical Insertion**: New language sections inserted in alphabetical order on multi-language pages without overwriting existing sections

## Setup & Running

### Prerequisites
- Python 3.8+
- Wiktionary bot credentials (username@botname and bot password)
- For AI example generation, one of:
  - A [Sarvam](https://www.sarvam.ai/) API key (default backend, `LLM_BACKEND=sarvam`), or
  - [Ollama](https://ollama.com/) with an instruct model pulled, e.g. `translategemma:27b` (`LLM_BACKEND=ollama`)
  - (Generation is optional — both UIs allow a manual example sentence that bypasses the LLM.)

### Installation
```bash
pip install -r requirements.txt
```

### Environment Variables
Create `.env` in project root:
```
WIKTIONARY_USERNAME="YourUsername@YourBotName"
WIKTIONARY_PASSWORD="your_generated_bot_password"
YOUR_EMAIL="your.email@example.com"

# LLM backend: "sarvam" (hosted, default) or "ollama" (local fallback)
LLM_BACKEND="sarvam"
SARVAM_API_KEY="your_sarvam_key"
SARVAM_MODEL="sarvam-30b"
OLLAMA_DRAFTER_MODEL="translategemma:27b"
```

### Running Applications

**Single-entry UI** (draft one word at a time):
```bash
streamlit run app.py
```
Opens at `http://localhost:8501`

**Bulk processing** (stage a batch from Alar YAML):
```bash
python bulk_processor.py
```
Generates up to 50 entries in `staged_entries.json`, skipping words that already exist on Wiktionary or are already staged/discarded.

**Bulk review UI** (review and publish staged entries):
```bash
streamlit run bulk_review_ui.py
```
Opens at `http://localhost:8501`, displays pending queue, allows manual/AI sentence generation, publishes with one click.

## Important Implementation Details

### Etymology Extraction (`core/scraper.py`)
The scraper targets DSAL's Kittel Dictionary API at `https://dsal.uchicago.edu/cgi-bin/app/kittel_query.py`. Key logic:
- If word has Sanskrit orthography (ಖ, ಘ, ಛ, ಠ, ಢ, ಥ, ಧ, ಫ, ಭ, ಶ, ಷ, ಋ, ಃ), assumes borrowing
- Matches markers: `Tbh.`/`Tatsama` → `{{bor|kn|sa|...}}`, `Ts.` → `{{bor|kn|sa|...}}`, `H.` → `{{bor|kn|hi|}}`
- If "no results" or no exact headword match, falls back to native Dravidian: `{{inh|kn|dra-pro|}}`
- Extracts cognates by regex pattern matching (`m.`, `t.`, `te.`, `tu.`, `mhr.` abbreviations) and rebuilds them in native scripts

### Morphology Selection (`core/linguistics.py`)
Analyzes word's final character:
- **Nouns ending in ು**: `{{kn-decl-u|word|stem}}`
- **Nouns ending in ಿ, ೆ, ೈ**: `{{kn-decl-e-i-ai|word}}`
- **Other nouns**: `{{kn-decl-a|word}}`
- **Verbs ending in ಕೊಳ್ಳು**: `{{kn-conj-koḷḷu|prefix}}`
- **Verbs ending in ಿಸು**: `{{kn-conj-isu|word|stem}}`
- **Verbs ending in ಿ, ೆ, ೈ**: `{{kn-conj-e-i-other|...}}`
- **Other verbs**: Marked as `IRREGULAR_CHECK` and manually expanded in UI

### LLM Generation (`app.py`, `bulk_review_ui.py`)
Single-stage generation through `core/llm_service.stream_chat()`:
- One **drafter** call (with `DRAFTER_PROMPT` + few-shot examples) generates a Kannada sentence containing the target word and its English translation.
- The backend is chosen at runtime by `LLM_BACKEND`: `sarvam` posts to Sarvam's OpenAI-compatible `/v1/chat/completions` (model from `SARVAM_MODEL`); `ollama` calls the local model from `OLLAMA_DRAFTER_MODEL`. Both stream text deltas.
- Output parsed via `parse_kannada_english()`: extracts text after `KANNADA:` and `ENGLISH:` labels.
- The earlier two-stage design (separate logic-auditor model) was removed in favor of a single, stronger model.

### Wiktionary Publishing (`core/wiktionary_api.py`)
1. Obtains login token
2. Authenticates with bot credentials
3. Fetches CSRF token
4. Retrieves existing page wikitext (if entry exists)
5. Calls `insert_alphabetically()` to merge new Kannada section in correct position
6. Publishes via `action=edit` with summary "Added Kannada entry via KannadaWiktionaryGenerator"
7. Returns success/failure and constructs URL: `https://en.wiktionary.org/wiki/{word}`

### Wikitext Template
Standard entry structure generated by both `app.py` and `bulk_processor.py`:
```
==Kannada==

===Etymology===
[Etymology line from scraper]. [Cognates if found].

===Pronunciation===
* {{kn-IPA|word}}

==={POS}===
{{kn-{pos_template}}}
# Definition 1
# Definition 2
#: {{ux|kn|Kannada sentence|tr=Transliteration|t=English translation}}

====[Morphology]====
{{kn-decl-... or kn-conj-...}}

===References===
* {{R:kn:Alar}}
* {{R:kn:Kittel}}
```

## Development Considerations

### Adding New Morphology Templates
1. Identify the word ending pattern in `core/linguistics.py` within `get_dynamic_morphology_block()`
2. Add new elif branch before the "IRREGULAR_CHECK" fallback
3. Test with words matching that pattern in both UIs

### Supporting New Cognate Languages
1. Add language abbreviation and code mapping to `lang_map` dict in `core/scraper.py`
2. Implement transliteration logic in `transliterate_cognate()` function using Unicode offset math
3. Test with words from Kittel that reference the new language

### Bulk Processor Scaling
- `BATCH_SIZE` constant in `bulk_processor.py` limits words processed per run (default: 50)
- Increase to process larger batches, but be aware of Wiktionary API rate limits
- Script includes 1-second sleep between API calls to Wiktionary

### Debugging Etymology Scraper
- Scraper outputs detailed logs to Streamlit UI under "Etymology Search Notes"
- Each log entry shows the API URL tested and reasoning for template selection
- Use logs to debug cognate extraction or marker-matching issues

## Common Tasks

**Process a batch of words:**
```bash
python bulk_processor.py && streamlit run bulk_review_ui.py
```

**Manually add a word without Ollama:**
```bash
streamlit run app.py
# Enter word, translation, POS
# Click "Fetch Etymology"
# Provide manual example sentence in "Manual Example Sentence" fields
# Skip Ollama generation step
# Review and publish
```

**Re-process a discarded word:**
Edit `data/discarded_entries.json` to remove the word, re-run `bulk_processor.py`.

**Check if word exists on Wiktionary before processing:**
Called automatically in both UIs via `check_wiktionary_entry_exists()`.

## Testing Strategy

- **Manual testing**: Run Streamlit UIs with test words, verify wikitext output
- **API testing**: `check_wiktionary_entry_exists()` can be called standalone against live Wiktionary
- **Scraper testing**: Test `source_etymology()` with various word patterns (Sanskrit-heavy, native, known cognates)
- **Transliteration testing**: Verify `transliterate_kannada_to_iso()` output against ISO 15919 standard


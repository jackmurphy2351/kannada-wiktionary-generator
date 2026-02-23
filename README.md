# Kannada Wiktionary Generator

A Streamlit-based tool for generating high-quality Wiktionary entries for Kannada words using LLMs (Qwen 2.5) and a "Ground Truth" verification system.

## Features
- **Deterministic Templates**: Automatic selection of `kn-decl-u`, `kn-decl-a`, and `kn-conj-u-irreg` based on linguistic rules.
- **Ground Truth Storage**: Saves verified entries to `verified_kannada_entries.json` to improve accuracy of future entries.
- **Sentence Sandbox**: Generates simple, SOV-structured usage examples.

## Setup
1. Install dependencies: `pip install streamlit ollama python-dotenv`
2. Ensure Ollama is running locally with the `qwen2.5:7b` model.
3. Run the app: `streamlit run app.py`
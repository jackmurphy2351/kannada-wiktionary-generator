# 🟨🟥 Kannada Wiktionary Generator 🟥🟨
The **Kannada Wiktionary Generator** is a specialized, modular pipeline designed to assist lexicographers and language enthusiasts in creating high-quality, standardized Wiktionary entries for Kannada words.

By combining web scraping, deterministic linguistic rules, Unicode manipulation, and an optional "Few-Shot" Local LLM approach, the app ensures that generated entries adhere to complex Wikitext formatting while maintaining strict etymological and grammatical accuracy.

## 🛠️ Key Features
1. Automated Etymology & Cognate Engine The app scrapes the University of Chicago's Digital South Asia Library (DSAL) Kittel Dictionary:

* **Marker Extraction**: Intelligently parses Kittel's specific abbreviations (e.g., `Tbh.`, `Ts.`, `H.`) to identify Tatsama, Tadbhava, and Hindustani loanwords.
* **Sanskrit Fallback**: Uses an orthographical heuristic to identify aspirated consonants and specific sibilants (like ಭ, ಧ, ಷ) to catch unmarked Sanskrit borrowings.
* **Cognate Script Shifting**: Identifies Dravidian sister-language cognates (Malayalam, Tamil, Telugu, Tulu) and uses Unicode offset mathematics to automatically translate Kittel's Kannada-script cognates into their native Indic scripts (e.g., converting 'ತಲೆ' to 'తల' for Telugu).

2. Deterministic Morphology Selection Identifies the correct Wiktionary morphology templates by analyzing word endings:
* **Nouns**: Automatically chooses between `kn-decl-u`, `kn-decl-e-i-ai`, or `kn-decl-a` based on the final vowel character.
* **Verbs**: Identifies reflexive forms (-ಕೊಳ್ಳು), causative forms (-ಿಸು), or regular endings to apply the appropriate conjugation template.

3. Dual-Mode Workflows (Single & Bulk Processing)
* **Single-Entry UI**: A streamlined interface for drafting a single word manually or bypassing the AI entirely.
* **Bulk Processing Pipeline**: Automatically parses thousands of words from the Alar open-source dictionary, filters out words that already exist on Wiktionary, and generates an MVP (Minimum Viable Product) Wikitext block for rapid staging.
* **Human-in-the-Loop Review**: A dedicated bulk-review UI to safely audit auto-generated entries, generate optional example sentences via Ollama, and publish them with a single click.

4. **Automatic Transliteration & Bolding**: A custom Python engine that deterministically maps Kannada text to the standard ISO 15919 transliteration format required by Wiktionary, while automatically applying correct Wikitext bolding syntax to target words.

5. **Direct Wiktionary Publishing**: Integrates directly with the MediaWiki API to streamline the publishing process:
* Checks if an entry already exists before drafting.
* Safely parses existing multi-language pages to insert the generated Kannada section in the correct alphabetical order (e.g., before Tulu) without overwriting existing linguistic data.

## 🧩 Modular Pipeline Structure
The codebase is structured to separate core linguistic logic from the execution scripts:

```Plaintext
kannada-wiktionary-generator/
├── app.py                # Single-entry Streamlit UI
├── bulk_processor.py     # Headless script for batch processing Alar YAML data
├── bulk_review_ui.py     # Streamlit UI for reviewing and publishing staged batches
├── core/                 # Core linguistic and API modules
│   ├── linguistics.py    
│   ├── wiktionary_api.py 
│   ├── scraper.py
│   ├── llm_service.py    
│   └── data_manager.py   
├── data/
│   ├── alar.yml          # Excluded from version control; must be downloaded
│   ├── staged_entries.json
│   └── verified_kannada_entries.json
└── requirements.txt      
```

## ⚙️ Workflow
### For Bulk Processing:
1. Run `python bulk_processor.py` to ingest a batch of words from the Alar database, fetch their etymologies, and stage the Wikitext MVP.
2. Run `streamlit run bulk_review_ui.py` to open the review queue.
3. Generate or manually add example sentences, verify the logic, and hit "Publish directly to Wiktionary".

### For Single Entries:
Run `streamlit run app.py` to draft a highly specific or missing word one at a time.

## 🚀 Setup & Installation
### Prerequisites
* Python 3.8+
* A Wiktionary Bot Password for automated publishing.
* (Optional) [Ollama](https://ollama.com/) installed and running locally with the `translategemma:4b` model pulled (`ollama pull translategemma:4b`) if you wish to use the AI generation feature.

### Installation
1. **Clone the repository**:
```Bash
git clone [your-repo-url]
cd kannada-wiktionary-generator
```

2. **Install Python dependencies**:
```Bash
pip install -r requirements.txt
```

3. **Fetch the Alar Dictionary Data**:
Download the latest `alar.yml` dataset from the [Alar Dictionary GitHub Repository](https://github.com/alar-dict/data) and place it inside the `/data/` directory.

4. **Environment Variables**
Create a `.env` file in the root directory to store your Wiktionary credentials safely:
```Plaintext
WIKTIONARY_USERNAME="YourUsername@YourBotName"
WIKTIONARY_PASSWORD="your_generated_bot_password"
```

## 💬 Feedback & Support
If you encounter any bugs, have questions about the linguistic logic, or would like to suggest new features, please do not hesitate to reach out!
The preferred method for feedback is via **GitHub Issues**.

* Report a Bug: If the scraper fails or the wikitext formatting is incorrect, please open a new issue with the target word and a description of the error.
* Feature Requests: Have an idea for a new morphology template or another Indic script? Submit a feature request in the issues tab.
* Contributions: Pull requests are welcome! Please ensure any changes to the linguistic logic are documented with examples.

## 📜 License
This project is licensed under the MIT License.
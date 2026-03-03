# 🟨🟥 Kannada Wiktionary Generator 🟥🟨

The **Kannada Wiktionary Generator** is a specialized, modular pipeline designed to assist lexicographers and language enthusiasts in creating high-quality, standardized Wiktionary entries for Kannada words. 

By combining web scraping, deterministic linguistic rules, Unicode manipulation, and an optional "Few-Shot" Local LLM approach, the app ensures that generated entries adhere to complex Wikitext formatting while maintaining strict etymological and grammatical accuracy.

## 🛠️ Key Features

**1. Automated Etymology & Cognate Engine** The app scrapes the University of Chicago's Digital South Asia Library (DSAL) Kittel Dictionary:
* **Marker Extraction:** Intelligently parses Kittel's specific abbreviations (e.g., `Tbh.`, `Ts.`, `H.`) to identify Tatsama, Tadbhava, and Hindustani loanwords.
* **Sanskrit Fallback:** Uses an orthographical heuristic to identify aspirated consonants and specific sibilants (like ಭ, ಧ, ಷ) to catch unmarked Sanskrit borrowings.
* **Cognate Script Shifting:** Identifies Dravidian sister-language cognates (Malayalam, Tamil, Telugu, Tulu) and uses Unicode offset mathematics to automatically translate Kittel's Kannada-script cognates into their native Indic scripts (e.g., converting 'ತಲೆ' to 'తల' for Telugu).

**2. Deterministic Morphology Selection** Identifies the correct Wiktionary morphology templates by analyzing word endings:
* **Nouns:** Automatically chooses between `kn-decl-u`, `kn-decl-e-i-ai`, or `kn-decl-a` based on the final vowel character.
* **Verbs:** Identifies reflexive forms (`-ಕೊಳ್ಳು`), causative forms (`-ಿಸು`), or regular endings to apply the appropriate conjugation template.

**3. Manual Example Entry & Optional AI Generation** The default workflow assumes users will input their own pre-written Kannada sentence and English translation. However, if needed, the app features an **optional** AI bypass that generates simple Subject-Object-Verb (SOV) Kannada example sentences using a dual-agent LLM pipeline (a Drafter and a Logic Auditor) powered by `translategemma:27b`.

**4. Automatic Transliteration** Includes a custom Python engine that deterministically maps Kannada text to the standard ISO 15919 transliteration format required by Wiktionary.

**5. Direct Wiktionary Publishing** Integrates directly with the MediaWiki API to streamline the publishing process:
* Checks if an entry already exists before drafting.
* Safely parses existing multi-language pages to insert the generated Kannada section in the correct alphabetical order (e.g., before Tulu) without overwriting existing linguistic data.

**6. Few-Shot Ground Truth System** Maintains a local `verified_kannada_entries.json` database to store finalized entries and seamlessly feed high-quality examples into the optional LLM prompt.

## 🧩 Modular Pipeline Structure

The codebase is split into dedicated modules to ensure clean separation of concerns and easy debugging:
* `app.py` - The main Streamlit UI connecting all components.
* `scraper.py` - Web scraping logic for the DSAL Kittel Dictionary.
* `linguistics.py` - Deterministic rules for morphology, cognate script shifting, and ISO transliteration.
* `wiktionary_api.py` - MediaWiki API authentication, fetching, and alphabetical injection logic.
* `llm_service.py` - Optional Ollama local model integration, prompts, and few-shot formatting.
* `data_manager.py` - Handles local JSON database operations.

## ⚙️ Workflow

The application operates in a strict pipeline to prevent data bleeding:
* **Step 1: Etymology Lookup.** Enter a word and fetch its roots and cognates from the DSAL Kittel database. Review the audit logs to see exactly which tags or patterns triggered the result.
* **Step 2: Entry Assembly.** Once the etymology is confirmed, input a manual example sentence (or rely on the optional local AI), and click "Generate" to stitch the final Wikitext entry together. 
* **Step 3: Publish.** Verify the output, save it to the local database, and hit "Publish directly to Wiktionary" to automatically push the edit live via the MediaWiki API.

## 🚀 Setup & Installation

### Prerequisites
* Python 3.8+
* A Wiktionary **Bot Password** for automated publishing.
* *(Optional)* [Ollama](https://ollama.com/) installed and running locally with the `translategemma:27b` model pulled (`ollama pull translategemma:27b`) if you wish to use the AI generation feature.

### Installation
#### 1. Clone the repository:
```bash
git clone [your-repo-url]
cd kannada-wiktionary-generator
```

### 2. Install Python dependencies:
```bash
pip install -r requirements.txt
```
*(Ensure your requirements include streamlit, requests, beautifulsoup4, ollama, and python-dotenv)*

### 3. Environment Variables
Create a .env file in the root directory to store your Wiktionary credentials safely:
```
WIKTIONARY_USERNAME="YourUsername@YourBotName"
WIKTIONARY_PASSWORD="your_generated_bot_password"
```

## 💬 Feedback & Support
If you encounter any bugs, have questions about the linguistic logic, or would like to suggest new features, please do not hesitate to reach out! 
The preferred method for feedback is via **GitHub Issues**. 

* **Report a Bug:** If the scraper fails or the wikitext formatting is incorrect, please [open a new issue](https://github.com/jackmurphy2351/kannada-wiktionary-generator/issues) with the target word and a description of the error.
* **Feature Requests:** Have an idea for a new morphology template or another Indic script? Submit a feature request in the issues tab.
* **Contributions:** Pull requests are welcome! Please ensure any changes to the linguistic logic are documented with examples.

## 📜 License
This project is licensed under the MIT License.
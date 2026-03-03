# 🟨🟥 Kannada Wiktionary Generator 🟥🟨

The **Kannada Wiktionary Generator** is a specialized, semi-automated pipeline designed to assist lexicographers and language enthusiasts in creating high-quality, standardized Wiktionary entries for Kannada words. 

By combining web scraping, deterministic linguistic rules, Unicode manipulation, and a "Few-Shot" Local LLM approach, the app ensures that generated entries adhere to complex Wikitext formatting while maintaining strict etymological and grammatical accuracy.

## 🛠️ Key Features

**1. Automated Etymology & Cognate Engine** The app scrapes the University of Chicago's Digital South Asia Library (DSAL) Kittel Dictionary:
* **Marker Extraction:** Intelligently parses Kittel's specific abbreviations (e.g., `Tbh.`, `Ts.`, `H.`) to identify Tatsama, Tadbhava, and Hindustani loanwords.
* **Sanskrit Fallback:** Uses an orthographical heuristic to identify aspirated consonants and specific sibilants (like ಭ, ಧ, ಷ) to catch unmarked Sanskrit borrowings.
* **Cognate Script Shifting:** Identifies Dravidian sister-language cognates (Malayalam, Tamil, Telugu, Tulu) and uses Unicode offset mathematics to automatically translate Kittel's Kannada-script cognates into their native Indic scripts (e.g., converting 'ತಲೆ' to 'తల' for Telugu).

**2. Deterministic Morphology Selection** Identifies the correct Wiktionary morphology templates by analyzing word endings:
* **Nouns:** Automatically chooses between `kn-decl-u`, `kn-decl-e-i-ai`, or `kn-decl-a` based on the final vowel character.
* **Verbs:** Identifies reflexive forms (`-ಕೊಳ್ಳು`), causative forms (`-ಿಸು`), or regular endings to apply the appropriate conjugation template.

**3. AI Example Generation & Manual Bypass** Generates simple Subject-Object-Verb (SOV) Kannada example sentences using the `{{ux|kn|...}}` template. 
* **Ollama Integration:** Uses a dual-agent LLM pipeline (a Drafter and a Logic Auditor) powered by `translategemma:27b` to write and verify grammar.
* **Manual Override:** Allows users to input their own pre-written Kannada sentence and English translation, completely bypassing the AI generation step to save time and compute.

**4. Automatic Transliteration** Includes a custom Python engine that deterministically maps Kannada text to the standard ISO 15919 transliteration format required by Wiktionary.

**5. Few-Shot Ground Truth System** Maintains a local `verified_kannada_entries.json` database. The app automatically feeds previously verified, high-quality entries into the LLM prompt to ensure the model mimics standard Wikitext formatting perfectly.

## ⚙️ Workflow

The application operates in a strict, two-step pipeline to prevent data bleeding:
* **Step 1: Etymology Lookup.** Enter a word and fetch its roots and cognates from the DSAL Kittel database. Review the audit logs to see exactly which tags or patterns triggered the result.
* **Step 2: Entry Assembly.** Once the etymology is confirmed, provide an optional manual example sentence (or rely on the local AI), and click "Generate" to stitch the final Wikitext entry together. Verify the output and save it to the local database.

## 🚀 Setup & Installation

### Prerequisites
* Python 3.8+
* [Ollama](https://ollama.com/) installed and running locally.
* Download the required 27B model for drafting and auditing:
```bash
ollama pull translategemma:27b
```

### Installation
#### 1. Clone the repository:
```bash
git clone [your-repo-url]
cd kannada-wiktionary-generator
```
#### 2. Install Python dependencies:
```bash
pip install -r requirements.txt
```
*(Ensure your requirements include `streamlit`, `requests`, `beautifulsoup4`, `ollama`, and `python-dotenv`)*

## 📜 License
This project is licensed under the MIT License.

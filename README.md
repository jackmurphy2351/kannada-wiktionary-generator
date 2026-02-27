# 🌿 Kannada Wiktionary Generator
The **Kannada Wiktionary Generator** is a specialized tool designed to assist language learners and lexicographers in creating high-quality, standardized Wiktionary entries for Kannada words. By combining deterministic linguistic rules with a "Few-Shot" LLM approach, the app ensures that generated entries adhere to complex Wikitext formatting while maintaining high etymological and grammatical accuracy.

## 🎯 Project Goals
* **Learner-Centric:** Focus on generating entries with clear usage examples and SOV-structured sentences to aid Kannada learners.

* **Linguistic Consistency:** Automate the selection of declension and conjugation templates based on the grammatical properties of the target word.

* **Knowledge Persistence:** Maintain a "Ground Truth" of verified entries that act as a golden standard for future generations.

## 🧠 Model Selection
The application utilizes **Ollama** to run two distinct classes of local LLMs:

* **TranslateGemma (12B) - "Fast Mode":** Optimized for cross-lingual accuracy and strict template adherence. Recommended for straightforward nouns and regular verbs.

* **Sarvam-M (24B) - "Deep Mode":** A larger model focused on deep linguistic reasoning and etymological nuance. Recommended for complex etymologies and irregular morphology.

## 🛠️ Key Features
**1. Deterministic Morphology Selection**  
The app identifies the correct Wiktionary templates by analyzing word endings:

* **Nouns:** Automatically chooses between `kn-decl-u`, `kn-decl-e-i-ai`, or `kn-decl-a` based on the final vowel character.

* **Verbs:** Identifies reflexive forms (`-ಕೊಳ್ಳು`), causative forms (`-ಿಸು`), or regular endings to apply the appropriate conjugation template.

**2. Few-Shot Ground Truth System**  
The application uses `verified_kannada_entries.json` as a local knowledge base. Before generating a new entry, the system:

1. Searches the JSON for words with matching parts of speech or structural templates.
2. Feeds these "golden entries" into the LLM prompt as context.
3. Ensures the model mimics the formatting of previously verified data.

**3. Sentence Sandbox**  
A dedicated module for generating three simple Subject-Object-Verb (SOV) Kannada sentences using the `{{ux|kn|...}}` template, allowing users to verify usage before saving to the ground truth.

📂 Project Structure
* `app.py`: The core Streamlit application containing the UI, linguistic logic, and Ollama integration.

* `verified_kannada_entries.json`: The storage file for all high-quality, user-verified Wikitext entries.

* `requirements.txt`: Python dependencies.

* `LICENSE`: MIT License.

## 🚀 Setup & Installation
### Prerequisites
* [Ollama](https://ollama.com/) installed and running locally.

* Download the required models:

```Bash
ollama pull translategemma:12b
ollama pull mashriram/sarvam-m:latest
```
### Installation
**1. Clone the repository:**

```Bash
git clone [your-repo-url]
cd kannada-wiktionary-generator
```
**2. Install Python dependencies:**

```Bash
pip install -r requirements.txt
```

**3. Environment Setup:**
Create a `.env` file for any local environment variables (if required).

**4. Run the Application:**

```Bash
streamlit run app.py
```

## 📜 License
This project is licensed under the MIT License.
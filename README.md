# Dataset-Conditioned Text-to-Image Generation Model

This project builds a **Text-to-Image Generation System** that utilizes your local dataset of **328 concept images across 21 categories** as visual examples and conditioning guidance for generative image models.

---

## 📁 Project Architecture

```
text to image/
│
├── api_key.txt             # 🔑 Ultra-simple file: paste your API key here directly
├── .env                    # 🔑 Environment variables file for provider API keys
├── .env.example            # 📄 Example template for environment keys
├── api_key.py              # ⚙️ Dedicated API key manager module
├── requirements.txt        # 📦 Required Python dependencies
│
├── dataset_loader.py       # 🔍 Indexes 328 images & retrieves examples matching prompts
├── model_generator.py      # 🤖 Multi-provider generation model conditioned on dataset examples
├── generate.py             # 💻 Command-line interface (CLI) for prompt-to-image generation
├── app.py                  # 🎨 Streamlit Web UI with interactive visual explorer
│
├── choices.json            # 📋 Concept-to-image dataset mappings
├── images/                 # 🖼️ Dataset folders (animals, fruits, nature, transportation, etc.)
└── outputs/                # 💾 Generated images and previews saved here
```

---

## 🔑 How to Provide Your API Key

You can provide your API key later using any of these 3 easy methods:

### Method 1: Paste directly into `api_key.txt` (Fastest & Simplest)
Open `api_key.txt` in any text editor and paste your key:
```text
AIzaSy...YourKeyHere
```
*(The system automatically detects whether it is a Google Gemini, OpenAI, Stability, or Hugging Face key!)*

### Method 2: Edit `.env`
Open `.env` and fill in the corresponding line:
```env
# Google Gemini / Imagen 3 (https://aistudio.google.com/)
GEMINI_API_KEY=AIzaSy...

# OpenAI DALL-E 3 (https://platform.openai.com/api-keys)
OPENAI_API_KEY=sk-...

# Stability AI (https://platform.stability.ai/)
STABILITY_API_KEY=sk-...

# Hugging Face Token (https://huggingface.co/settings/tokens)
HF_TOKEN=hf_...
```

### Method 3: Via CLI or Web Interface
- Run:
  ```bash
  python generate.py --set-key YOUR_KEY --provider gemini
  ```
- Or enter it directly in the **Streamlit Web UI sidebar**.

---

## 📦 Requirements & Installation

All required packages are specified in `requirements.txt`. Install with:

```bash
pip install -r requirements.txt
```

*(Note: Essential packages like `requests`, `pillow`, `google-generativeai`, `openai`, `diffusers`, `torch`, and `streamlit` are already installed in your current Python environment.)*

---

## 🧠 How the Model Uses Dataset Images as Examples

1. **Concept Extraction**:
   When you input a prompt (e.g. `"a cute playful dog running across green grass with a bright red ball"`), `dataset_loader.py` tokenizes the text into n-grams (1, 2, and 3-word phrases).
2. **Visual Reference Retrieval**:
   The engine matches concepts against the 328 dataset images (e.g., matching `'dog'` from `images/animals/dog.jpg`, `'running'` from `images/sports/running.jpg`, `'green'` from `images/colors/green.jpg`).
3. **In-Context Conditioning**:
   `model_generator.py` extracts the visual aesthetics of the matched references (clear subject isolation, natural lighting, true-to-life textures, clean framing) and generates an enhanced conditioning prompt.
4. **Generation & Fallback Preview**:
   - When an API key is provided, it calls the cloud model (e.g., Google Imagen 3, DALL-E 3, or Stability AI) with the conditioned prompt.
   - If an API key is not yet set, it produces a **Dataset Reference Preview Composite** showing the prompt, matched example images, and style guidance without failing or throwing errors.

---

## 🚀 Usage

### 1. Command-Line Generation (`generate.py`)

#### Generate an image:
```bash
python generate.py --prompt "a smiling doctor in a clinic holding a notebook"
```

#### Choose a specific provider (Gemini, OpenAI, Stability, HuggingFace):
```bash
python generate.py --prompt "a fresh mango and coconut on a wooden table" --provider gemini
```

#### Specify custom output filename:
```bash
python generate.py --prompt "a red sports car under a rainbow" --output red_car.jpg
```

#### Check API key status:
```bash
python generate.py --check-keys
```

#### Browse dataset categories and concepts:
```bash
python generate.py --browse-dataset
```

---

### 2. Interactive Web Application (`app.py`)

Launch the web app in your browser:
```bash
streamlit run app.py
```

The Web UI features:
- **Live Concept Detection**: As you type your prompt, matched dataset reference thumbnails appear in real time.
- **Dataset Visual Explorer**: Browse through all 328 images across 21 categories in an image gallery.
- **Side-by-Side Comparison**: View the dataset reference examples alongside the final generated image.
- **One-Click Key Management**: Add or switch API keys directly from the sidebar.


"""
Interactive Web Application for Dataset-Conditioned Text-to-Image Generation
=============================================================================
Run with:
    streamlit run app.py
"""

import os
from pathlib import Path
import streamlit as st
from PIL import Image

from api_key import get_status_info, get_api_key, save_api_key, PROVIDER_NAMES, PROVIDER_ENV_VARS
from dataset_loader import get_dataset
from model_generator import get_model, OUTPUTS_DIR

st.set_page_config(
    page_title="Text-to-Image Model with Dataset Conditioning",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling
st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        color: #64748b;
        font-size: 1.05rem;
        margin-bottom: 1.5rem;
    }
    .badge-pill {
        display: inline-block;
        padding: 0.25rem 0.6rem;
        border-radius: 9999px;
        font-size: 0.8rem;
        font-weight: 600;
        background-color: #e0e7ff;
        color: #3730a3;
        margin-right: 0.5rem;
    }
    .card-box {
        border-radius: 8px;
        padding: 1rem;
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        margin-bottom: 1rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_resources():
    ds = get_dataset()
    model = get_model()
    return ds, model


dataset, model = load_resources()

# ==============================================================================
# SIDEBAR: API KEY & MODEL SETTINGS
# ==============================================================================
with st.sidebar:
    st.header("⚙️ Model & API Settings")

    key_status = get_status_info()

    # Provider Selection
    provider_options = list(PROVIDER_NAMES.keys())
    provider_labels = [PROVIDER_NAMES[p] for p in provider_options]

    default_idx = 0
    if key_status["active_provider"] in provider_options:
        default_idx = provider_options.index(key_status["active_provider"])

    selected_label = st.selectbox(
        "Choose Generation Engine:",
        options=provider_labels,
        index=default_idx,
    )
    selected_provider = provider_options[provider_labels.index(selected_label)]

    # Check key for selected provider
    current_key, _ = get_api_key(selected_provider)

    st.markdown("---")
    st.subheader("🔑 API Key Status")

    if current_key:
        st.success(f"Key Ready for {selected_label}")
        st.caption(f"Preview: `{current_key[:6]}...{current_key[-4:]}`")
    else:
        st.warning(f"No key yet for {selected_label}")
        st.info(
            "You can provide your key in **`api_key.txt`**, in **`.env`**, "
            "or enter it directly below to save it."
        )

    # Input to set or update API key
    with st.expander("Update / Enter API Key", expanded=(not current_key)):
        new_key_input = st.text_input(
            f"Enter {selected_label} Key:",
            type="password",
            placeholder="Paste your key here...",
            help="This will be saved to api_key.txt and .env for future runs.",
        )
        if st.button("Save Key", type="primary"):
            if new_key_input.strip():
                save_api_key(new_key_input.strip(), provider=selected_provider)
                st.success(f"Key saved for {selected_label}! Reloading...")
                st.rerun()
            else:
                st.error("Please paste a valid key string.")

    st.markdown("---")
    st.subheader("📊 Dataset Statistics")
    st.write(f"📁 **Total Images:** {dataset.total_images()}")
    st.write(f"🏷️ **Categories ({len(dataset.get_categories())}):**")
    st.caption(", ".join(dataset.get_categories()))


# ==============================================================================
# MAIN PAGE
# ==============================================================================
st.markdown('<div class="main-title">🎨 Dataset-Conditioned Text-to-Image Generation</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">Uses your local image dataset as visual reference examples and few-shot conditioning to generate high-fidelity AI images.</div>',
    unsafe_allow_html=True,
)

# Tabs: 1. Generate, 2. Dataset Explorer, 3. Setup Guide
tab_gen, tab_explore, tab_guide = st.tabs(["🚀 Generate Image", "📚 Browse Dataset Examples", "📖 API Setup Guide"])

with tab_gen:
    col_input, col_meta = st.columns([2, 1])

    with col_input:
        prompt_input = st.text_area(
            "Enter your image prompt:",
            value="a cute playful puppy running across green grass with a bright red ball",
            height=100,
            help="Describe the scene. Any concept present in the dataset (like 'dog', 'running', 'green', 'ball') will be extracted as visual reference.",
        )

        sample_prompts = [
            "a smiling doctor standing in a modern clinic with a notebook",
            "a fresh ripe mango and coconut resting on a wooden table",
            "a sleek sports car on a wet street under rain and rainbow",
            "a cheerful child wearing a yellow hat and blue coat playing outdoors",
        ]

        st.caption("💡 Quick Prompt Presets:")
        preset_cols = st.columns(len(sample_prompts))
        for idx, sp in enumerate(sample_prompts):
            with preset_cols[idx]:
                if st.button(f"Preset {idx+1}", key=f"pre_{idx}", help=sp):
                    st.session_state["active_prompt"] = sp
                    st.rerun()

        if "active_prompt" in st.session_state:
            prompt_input = st.session_state.pop("active_prompt")

    # Match dataset examples live
    matched_examples = dataset.match_prompt(prompt_input, top_k=4)

    with col_meta:
        st.markdown("##### 🔍 Matched Dataset Concepts")
        if matched_examples:
            st.success(f"Matched {len(matched_examples)} reference concept(s) from dataset:")
            for m in matched_examples:
                st.markdown(f"- **{m.name.title()}** (`{m.category}`)")
        else:
            st.info("No direct keyword match in dataset. General generation will apply.")

    # Show Reference Thumbnails
    if matched_examples:
        st.markdown("##### 🖼️ Visual References Sent to Model")
        ex_cols = st.columns(len(matched_examples))
        for idx, m in enumerate(matched_examples):
            with ex_cols[idx]:
                try:
                    img = m.image
                    st.image(img, caption=f"{m.name.title()} ({m.category})", use_container_width=True)
                except Exception as e:
                    st.caption(f"Error loading {m.name}: {e}")

    st.markdown("---")

    # Generate Action
    btn_generate = st.button("✨ Generate Image with Model", type="primary", use_container_width=True)

    if btn_generate:
        if not prompt_input.strip():
            st.warning("Please provide a prompt first.")
        else:
            with st.spinner(f"Synthesizing image with {selected_label}..."):
                res = model.generate(
                    prompt=prompt_input,
                    provider=selected_provider,
                )

                if res["success"]:
                    st.success("Generation completed!")

                    res_col1, res_col2 = st.columns([3, 2])

                    with res_col1:
                        st.subheader("🖼️ Generated Image")
                        st.image(res["image"], use_container_width=True)

                        if res["saved_path"]:
                            st.caption(f"Saved locally to: `{res['saved_path']}`")

                    with res_col2:
                        st.subheader("📋 Conditioning & Generation Details")
                        st.write(f"**Provider:** `{res['provider']}`")
                        st.write(f"**Target Prompt:** {res['original_prompt']}")

                        with st.expander("Show Synthesized Conditioned Prompt", expanded=True):
                            st.text(res["conditioned_prompt"])

                        if res["error"]:
                            st.info(f"ℹ️ **Notice:** {res['error']}")
                else:
                    st.error(f"Generation failed: {res['error']}")


with tab_explore:
    st.subheader("📚 Dataset Visual Explorer")
    st.write(
        f"Browse the **328 reference images** categorized across 21 domain folders. "
        "These images form the visual foundation and examples used to condition the model."
    )

    cats = dataset.get_categories()
    selected_cat = st.selectbox("Select a Category to Explore:", options=cats)

    items_in_cat = dataset.get_items_by_category(selected_cat)
    st.write(f"Showing **{len(items_in_cat)}** concepts in category **'{selected_cat}'**:")

    # Grid display
    cols_per_row = 6
    for i in range(0, len(items_in_cat), cols_per_row):
        row_items = items_in_cat[i : i + cols_per_row]
        cols = st.columns(len(row_items))
        for col, it in zip(cols, row_items):
            with col:
                try:
                    st.image(it.image, caption=it.name.title(), use_container_width=True)
                except Exception as e:
                    st.caption(f"{it.name}: error")


with tab_guide:
    st.subheader("📖 API Key Setup & Configuration Guide")
    st.markdown(
        """
        You can configure your API key through any of the three easy methods below:

        ### Method 1: Paste directly into `api_key.txt` (Easiest)
        1. Open the file **`api_key.txt`** located in this folder.
        2. Paste your API key on the line below the comments and save the file.
        3. That's it! The system will automatically detect and use your key.

        ### Method 2: Configure in `.env` File
        Open **`.env`** and paste your key next to your preferred provider:
        ```ini
        # Google Gemini / Imagen
        GEMINI_API_KEY=AIzaSyYourKeyHere

        # OpenAI DALL-E 3
        OPENAI_API_KEY=sk-proj-YourKeyHere

        # Stability AI
        STABILITY_API_KEY=sk-YourKeyHere

        # Hugging Face
        HF_TOKEN=hf_YourKeyHere
        ```

        ### Method 3: Use the Sidebar in this App
        Open the sidebar on the left, expand **Update / Enter API Key**, paste your key, and click **Save Key**.

        ---

        ### How the Model Uses Dataset Examples:
        1. **Concept Extraction:** When you enter a prompt, the tokenizer scans for known entities, actions, colors, and objects in `choices.json` and the `images/` directory.
        2. **Reference Retrieval:** Relevant dataset images are fetched and their visual properties (silhouette, isolated subject style, natural lighting, and color palette) are analyzed.
        3. **Prompt Conditioning:** The model constructs an enriched conditioning prompt embedding the visual style of your dataset references.
        4. **Image Generation:** The image generator produces high-resolution output reflecting both your text prompt and the visual identity of the examples.
        """
    )


"""
Model Generator Module
=======================
Text-to-Image Generation Engine conditioned on local dataset examples.

Supported Providers:
- Google Gemini / Imagen 3 (imagen-3.0-generate-002)
- OpenAI DALL-E 3
- Stability AI (Stable Diffusion 3 / Core)
- Hugging Face Inference API (FLUX.1 / SDXL)
- Local Demonstration / Fallback Mode (Runs when API key is not yet set)
"""

import base64
import io
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import requests
from PIL import Image, ImageDraw, ImageFont

from api_key import get_api_key, detect_provider_from_key, PROVIDER_NAMES
from dataset_loader import get_dataset, DatasetItem

BASE_DIR = Path(__file__).resolve().parent
OUTPUTS_DIR = BASE_DIR / "outputs"
OUTPUTS_DIR.mkdir(exist_ok=True)


class TextToImageModel:
    """
    Main Text-to-Image generation model that uses dataset examples
    for conditioning and style guidance.
    """

    def __init__(self, provider: Optional[str] = None):
        self.dataset = get_dataset()
        self.requested_provider = provider

    def get_effective_provider_and_key(self, provider: Optional[str] = None) -> Tuple[str, Optional[str]]:
        """Resolves the active provider and its API key."""
        target_p = provider or self.requested_provider
        key, resolved_p = get_api_key(target_p)
        return resolved_p, key

    def build_conditioned_prompt(
        self, user_prompt: str, example_items: List[DatasetItem]
    ) -> Tuple[str, str]:
        """
        Enhances the user's prompt by conditioning it on the matched dataset examples.
        
        Returns:
            (enhanced_prompt, conditioning_notes)
        """
        if not example_items:
            enhanced = (
                f"{user_prompt}, high resolution, sharp focus, professional composition, "
                f"vibrant colors, clean lighting, 8k quality photorealistic render."
            )
            notes = "Standard prompt enhancement (no specific dataset references found)."
            return enhanced, notes

        # Extract concept details from dataset items
        concepts = [f"'{item.name}' ({item.category})" for item in example_items]
        concept_str = ", ".join(concepts)

        # Style cues matching the educational/photographic dataset
        style_cues = (
            "clean subject isolation, authentic texture, natural studio lighting, "
            "vivid true-to-life colors, accurate anatomy and material properties"
        )

        enhanced = (
            f"{user_prompt}. Visual references: consistent with {concept_str} dataset examples, "
            f"featuring {style_cues}. Professional photography, hyper-detailed, clean 4k composition."
        )

        notes = (
            f"Prompt conditioned using {len(example_items)} dataset reference example(s): {concept_str}.\n"
            f"Visual style cues applied: {style_cues}."
        )
        return enhanced, notes

    def generate(
        self,
        prompt: str,
        provider: Optional[str] = None,
        custom_examples: Optional[List[DatasetItem]] = None,
        save_output: bool = True,
        output_filename: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Generates an image from prompt, incorporating dataset examples as references.
        
        Args:
            prompt: User text prompt
            provider: 'gemini', 'openai', 'stability', 'huggingface', or None (auto)
            custom_examples: Explicit dataset items to use as reference (optional)
            save_output: Whether to automatically save the generated image to outputs/
            output_filename: Optional specific filename
        
        Returns:
            Dict containing generation result, image object, metadata, and status.
        """
        active_provider, api_key = self.get_effective_provider_and_key(provider)

        # 1. Retrieve dataset examples matching prompt
        if custom_examples is not None:
            examples = custom_examples
        else:
            examples = self.dataset.match_prompt(prompt, top_k=3)

        # 2. Build conditioned prompt
        conditioned_prompt, conditioning_notes = self.build_conditioned_prompt(prompt, examples)

        result = {
            "success": False,
            "image": None,
            "saved_path": None,
            "provider": active_provider,
            "provider_name": PROVIDER_NAMES.get(active_provider, active_provider),
            "original_prompt": prompt,
            "conditioned_prompt": conditioned_prompt,
            "examples": [ex.to_dict() for ex in examples],
            "conditioning_notes": conditioning_notes,
            "error": None,
        }

        # Check if API key is missing
        if not api_key:
            # Fallback to interactive demonstration composite
            demo_img = self._create_demo_reference_preview(prompt, examples, conditioned_prompt, active_provider)
            result["image"] = demo_img
            result["success"] = True
            result["provider"] = f"{active_provider} (Dataset Example Preview Mode)"
            result["error"] = (
                f"API key not yet provided for {PROVIDER_NAMES.get(active_provider, active_provider)}. "
                f"Generated preview visualization using matched dataset examples. "
                f"Please paste your API key in 'api_key.txt' or '.env' to enable full cloud AI image generation."
            )
            if save_output:
                saved_path = self._save_image(demo_img, prompt, output_filename, prefix="preview")
                result["saved_path"] = str(saved_path)
            return result

        # Execute generation with chosen provider
        try:
            if active_provider == "gemini":
                img = self._generate_gemini(conditioned_prompt, api_key)
            elif active_provider == "openai":
                img = self._generate_openai(conditioned_prompt, api_key)
            elif active_provider == "stability":
                img = self._generate_stability(conditioned_prompt, api_key, examples)
            elif active_provider == "huggingface":
                img = self._generate_huggingface(conditioned_prompt, api_key)
            else:
                img = self._generate_gemini(conditioned_prompt, api_key)

            result["image"] = img
            result["success"] = True

            if save_output:
                saved_path = self._save_image(img, prompt, output_filename)
                result["saved_path"] = str(saved_path)

        except Exception as e:
            result["success"] = False
            result["error"] = str(e)
            print(f"[Generation Error] {e}")

        return result

    # --------------------------------------------------------------------------
    # Provider Implementations
    # --------------------------------------------------------------------------

    def _generate_gemini(self, prompt: str, api_key: str) -> Image.Image:
        """
        Generates image using Google's Imagen 3 API.
        Endpoint: imagen-3.0-generate-002:predict
        """
        url = f"https://generativelanguage.googleapis.com/v1beta/models/imagen-3.0-generate-002:predict?key={api_key}"
        payload = {
            "instances": [{"prompt": prompt}],
            "parameters": {
                "sampleCount": 1,
                "aspectRatio": "1:1",
                "outputMimeType": "image/jpeg",
                "personGeneration": "ALLOW_ADULT",
            },
        }
        headers = {"Content-Type": "application/json"}

        response = requests.post(url, json=payload, headers=headers, timeout=60)

        # Fallback to fast model if model quota or availability requires it
        if response.status_code == 404:
            url_fast = f"https://generativelanguage.googleapis.com/v1beta/models/imagen-3.0-fast-generate-001:predict?key={api_key}"
            response = requests.post(url_fast, json=payload, headers=headers, timeout=60)

        if response.status_code != 200:
            error_data = response.json().get("error", {})
            msg = error_data.get("message", response.text)
            raise RuntimeError(f"Google Imagen API Error ({response.status_code}): {msg}")

        data = response.json()
        predictions = data.get("predictions", [])
        if not predictions or "bytesBase64Encoded" not in predictions[0]:
            raise RuntimeError(f"No image returned by Imagen API: {data}")

        img_b64 = predictions[0]["bytesBase64Encoded"]
        img_bytes = base64.b64decode(img_b64)
        return Image.open(io.BytesIO(img_bytes)).convert("RGB")

    def _generate_openai(self, prompt: str, api_key: str) -> Image.Image:
        """
        Generates image using OpenAI's image generation models (gpt-image-1 / gpt-image-1.5).
        """
        from openai import OpenAI

        client = OpenAI(api_key=api_key)
        candidate_models = ["gpt-image-1", "gpt-image-1.5", "gpt-image-1-mini", "gpt-image-2", "chatgpt-image-latest"]
        
        last_error = None
        response = None

        for model_id in candidate_models:
            try:
                response = client.images.generate(
                    model=model_id,
                    prompt=prompt,
                    n=1,
                )
                if response and response.data:
                    break
            except Exception as e:
                last_error = e
                err_text = str(e).lower()
                # If quota exhausted, don't keep trying all models since all will fail with 429
                if "insufficient_quota" in err_text or "credit_balance_exhausted" in err_text:
                    raise RuntimeError(
                        "OpenAI Quota Exhausted (429): Your OpenAI account has no remaining billing credits. "
                        "Please add credits at https://platform.openai.com/settings/organization/billing "
                        "or use a Google Gemini / Hugging Face key in .env or api_key.txt."
                    )
                continue

        if response is None or not response.data:
            raise RuntimeError(f"OpenAI Image Generation Error: {last_error}")

        img_item = response.data[0]
        if hasattr(img_item, "b64_json") and img_item.b64_json:
            img_bytes = base64.b64decode(img_item.b64_json)
        elif hasattr(img_item, "url") and img_item.url:
            res = requests.get(img_item.url, timeout=60)
            if res.status_code != 200:
                raise RuntimeError(f"Failed to download generated image from OpenAI: {res.status_code}")
            img_bytes = res.content
        else:
            raise RuntimeError(f"Unexpected response structure from OpenAI: {response}")

        return Image.open(io.BytesIO(img_bytes)).convert("RGB")

    def _generate_stability(
        self, prompt: str, api_key: str, examples: List[DatasetItem]
    ) -> Image.Image:
        """
        Generates image using Stability AI Core / SD3 API.
        """
        url = "https://api.stability.ai/v2beta/stable-image/generate/core"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Accept": "image/*",
        }
        data = {
            "prompt": prompt,
            "output_format": "jpeg",
            "aspect_ratio": "1:1",
        }
        files = {}
        # If reference image is available, include it as style guide
        if examples and examples[0].path.exists():
            # Use structure or image-to-image if applicable
            pass

        response = requests.post(url, headers=headers, data=data, files=files, timeout=60)
        if response.status_code != 200:
            raise RuntimeError(f"Stability AI Error ({response.status_code}): {response.text}")

        return Image.open(io.BytesIO(response.content)).convert("RGB")

    def _generate_huggingface(self, prompt: str, api_key: str) -> Image.Image:
        """
        Generates image using Hugging Face Inference API (FLUX.1-schnell or SDXL).
        """
        model_id = "black-forest-labs/FLUX.1-schnell"
        url = f"https://api-inference.huggingface.co/models/{model_id}"
        headers = {"Authorization": f"Bearer {api_key}"}
        payload = {"inputs": prompt}

        response = requests.post(url, headers=headers, json=payload, timeout=60)
        if response.status_code != 200:
            # Fallback to SDXL
            url = "https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-xl-base-1.0"
            response = requests.post(url, headers=headers, json=payload, timeout=60)

        if response.status_code != 200:
            raise RuntimeError(f"Hugging Face API Error ({response.status_code}): {response.text}")

        return Image.open(io.BytesIO(response.content)).convert("RGB")

    # --------------------------------------------------------------------------
    # Demonstration / Preview Composite (When API Key is Pending)
    # --------------------------------------------------------------------------

    def _create_demo_reference_preview(
        self,
        prompt: str,
        examples: List[DatasetItem],
        conditioned_prompt: str,
        provider: str,
    ) -> Image.Image:
        """
        Creates an illustrative canvas showing the prompt, the conditioned guidance,
        and the matched dataset examples arranged side-by-side.
        """
        canvas_w, canvas_h = 1000, 600
        canvas = Image.new("RGB", (canvas_w, canvas_h), color=(20, 24, 33))
        draw = ImageDraw.Draw(canvas)

        # Draw decorative header banner
        draw.rectangle([(0, 0), (canvas_w, 80)], fill=(32, 42, 60))
        draw.text(
            (30, 20),
            "Dataset-Conditioned Text-to-Image Generation (Preview)",
            fill=(255, 255, 255),
        )
        draw.text(
            (30, 48),
            f"Provider Configured: {PROVIDER_NAMES.get(provider, provider)} | Status: Waiting for API Key in api_key.txt",
            fill=(200, 220, 255),
        )

        # Draw Prompt Info
        draw.rectangle([(30, 100), (canvas_w - 30, 180)], fill=(28, 35, 48), outline=(60, 75, 100))
        draw.text((45, 110), "Target User Prompt:", fill=(120, 200, 255))
        draw.text((45, 130), f'"{prompt}"', fill=(255, 255, 255))

        match_count = len(examples)
        draw.text(
            (45, 155),
            f"Matched Dataset Reference Examples: {match_count} image(s) found in local dataset",
            fill=(180, 240, 180) if match_count > 0 else (255, 180, 180),
        )

        # Render Example Images
        draw.text((30, 200), "Dataset Reference Images Used for Model Conditioning:", fill=(240, 240, 240))

        if examples:
            num_ex = min(len(examples), 4)
            card_w = (canvas_w - 60 - (num_ex - 1) * 20) // num_ex
            card_h = 280

            for idx, ex in enumerate(examples[:num_ex]):
                x0 = 30 + idx * (card_w + 20)
                y0 = 230
                # Draw card background
                draw.rectangle([(x0, y0), (x0 + card_w, y0 + card_h)], fill=(28, 35, 48), outline=(70, 85, 115))

                try:
                    img = ex.image
                    img.thumbnail((card_w - 20, card_h - 70))
                    # Center thumbnail
                    tx = x0 + (card_w - img.width) // 2
                    ty = y0 + 15 + (card_h - 70 - img.height) // 2
                    canvas.paste(img, (tx, ty))
                except Exception as e:
                    draw.text((x0 + 15, y0 + 50), f"[Image Load Error: {e}]", fill=(255, 100, 100))

                # Labels
                draw.rectangle([(x0, y0 + card_h - 45), (x0 + card_w, y0 + card_h)], fill=(38, 48, 68))
                draw.text((x0 + 10, y0 + card_h - 38), f"Concept: {ex.name.title()}", fill=(255, 255, 255))
                draw.text((x0 + 10, y0 + card_h - 20), f"Category: {ex.category}", fill=(180, 210, 240))
        else:
            draw.rectangle([(30, 230), (canvas_w - 30, 520)], fill=(28, 35, 48))
            draw.text(
                (50, 350),
                "No direct concept keyword matched in dataset for this prompt.\n"
                "The model will generate using pure descriptive diffusion once API key is provided.",
                fill=(200, 200, 200),
            )

        # Footer Notice
        draw.rectangle([(0, canvas_h - 45), (canvas_w, canvas_h)], fill=(15, 18, 26))
        draw.text(
            (30, canvas_h - 30),
            "To generate the final high-res AI image, enter your API key in api_key.txt or .env and run again.",
            fill=(255, 200, 100),
        )

        return canvas

    # --------------------------------------------------------------------------
    # Utility Helpers
    # --------------------------------------------------------------------------

    def _save_image(
        self,
        img: Image.Image,
        prompt: str,
        filename: Optional[str] = None,
        prefix: str = "generated",
    ) -> Path:
        """Saves PIL Image to the outputs folder."""
        if filename:
            out_file = OUTPUTS_DIR / filename
        else:
            safe_slug = "".join([c if c.isalnum() else "_" for c in prompt[:25]]).strip("_")
            timestamp = int(time.time())
            out_file = OUTPUTS_DIR / f"{prefix}_{safe_slug}_{timestamp}.jpg"

        img.save(out_file, format="JPEG", quality=95)
        return out_file


# Global helper instance
_MODEL_INSTANCE: Optional[TextToImageModel] = None


def get_model(provider: Optional[str] = None) -> TextToImageModel:
    global _MODEL_INSTANCE
    if _MODEL_INSTANCE is None or provider is not None:
        _MODEL_INSTANCE = TextToImageModel(provider=provider)
    return _MODEL_INSTANCE


if __name__ == "__main__":
    model = get_model()
    test_prompt = "a cute golden dog running happily on green grass"
    print(f"Testing generation pipeline for prompt: '{test_prompt}'")
    out = model.generate(test_prompt)
    print(f"Status: Success={out['success']}")
    print(f"Provider: {out['provider']}")
    print(f"Conditioned Prompt:\n{out['conditioned_prompt']}")
    if out["saved_path"]:
        print(f"Output saved to: {out['saved_path']}")
    if out["error"]:
        print(f"Notice: {out['error']}")


"""
Dataset Loader & Example Retriever Module
==========================================
Scans the local image dataset (`images/` folder and `choices.json`),
indexes all categories and concepts, and retrieves relevant visual examples
to guide the text-to-image generation model.
"""

import json
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from PIL import Image

BASE_DIR = Path(__file__).resolve().parent
IMAGES_DIR = BASE_DIR / "images"
CHOICES_FILE = BASE_DIR / "choices.json"

# Common spelling corrections / aliases present in dataset filenames
TYPO_CORRECTIONS = {
    "pegion": "pigeon",
    "squrril": "squirrel",
    "musk": "muskmelon",
    "scare": "scared",
    "chapthi": "chapati",
    "avocada": "avocado",
    "pomogranate": "pomegranate",
    "sky bkue": "sky blue",
    "leach": "lychee",
    "pineaPPLE": "pineapple",
}


class DatasetItem:
    """Represents a single reference image in the dataset."""

    def __init__(self, name: str, category: str, path: Path):
        self.name = name.lower()
        self.category = category.lower()
        self.path = path
        self.rel_path = str(path.relative_to(BASE_DIR)).replace("\\", "/")

    @property
    def image(self) -> Image.Image:
        """Loads and returns the PIL Image in RGB format."""
        img = Image.open(self.path)
        return img.convert("RGB")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "category": self.category,
            "path": str(self.path),
            "rel_path": self.rel_path,
        }

    def __repr__(self):
        return f"<DatasetItem: {self.category}/{self.name}>"


class ImageDataset:
    """
    Manages indexing and semantic keyword/category matching across the image dataset.
    """

    def __init__(self, images_dir: Path = IMAGES_DIR, choices_file: Path = CHOICES_FILE):
        self.images_dir = images_dir
        self.choices_file = choices_file
        self.items: Dict[str, DatasetItem] = {}  # key: unique identifier
        self.by_category: Dict[str, List[DatasetItem]] = {}
        self.keyword_index: Dict[str, DatasetItem] = {}

        self._build_index()

    def _build_index(self):
        """Scans the disk and choices.json to build a complete index."""
        # 1. Scan actual disk files
        if self.images_dir.exists():
            for root, _, files in os.walk(self.images_dir):
                root_path = Path(root)
                category = root_path.name
                if category == "images":
                    continue

                for file in files:
                    ext = file.lower().split(".")[-1]
                    if ext in ("jpg", "jpeg", "png", "webp"):
                        full_path = root_path / file
                        stem = full_path.stem.lower()

                        # Normalize typos
                        clean_name = TYPO_CORRECTIONS.get(stem, stem)
                        item_id = f"{category}/{clean_name}"
                        item = DatasetItem(name=clean_name, category=category, path=full_path)

                        self.items[item_id] = item
                        self.by_category.setdefault(category, []).append(item)
                        self.keyword_index[clean_name] = item

        # 2. Incorporate choices.json mappings
        if self.choices_file.exists():
            try:
                with open(self.choices_file, "r", encoding="utf-8") as f:
                    choices = json.load(f)
                for label, rel_p in choices.items():
                    label_clean = label.lower().strip()
                    p = BASE_DIR / rel_p
                    if p.exists():
                        cat = p.parent.name
                        item_id = f"{cat}/{label_clean}"
                        if item_id not in self.items:
                            item = DatasetItem(name=label_clean, category=cat, path=p)
                            self.items[item_id] = item
                            self.by_category.setdefault(cat, []).append(item)
                        self.keyword_index[label_clean] = self.items.get(item_id) or DatasetItem(label_clean, cat, p)
                    else:
                        # Attempt to resolve using filename stem
                        stem = Path(rel_p).stem.lower()
                        clean_stem = TYPO_CORRECTIONS.get(stem, stem)
                        if clean_stem in self.keyword_index:
                            self.keyword_index[label_clean] = self.keyword_index[clean_stem]
            except Exception as e:
                print(f"[Warning] Error parsing choices.json: {e}")

    def total_images(self) -> int:
        return len(self.items)

    def get_categories(self) -> List[str]:
        return sorted(list(self.by_category.keys()))

    def get_items_by_category(self, category: str) -> List[DatasetItem]:
        return self.by_category.get(category.lower(), [])

    def get_item(self, name_or_id: str) -> Optional[DatasetItem]:
        key = name_or_id.lower().strip()
        if key in self.items:
            return self.items[key]
        if key in self.keyword_index:
            return self.keyword_index[key]
        return None

    def match_prompt(self, prompt: str, top_k: int = 4) -> List[DatasetItem]:
        """
        Finds the most relevant dataset example images matching concepts in the prompt.
        Uses n-gram extraction (3-word, 2-word, single-word) and keyword matching.
        """
        if not prompt or not prompt.strip():
            return []

        cleaned = re.sub(r"[^\w\s]", " ", prompt.lower())
        words = cleaned.split()
        if not words:
            return []

        matched_items: List[DatasetItem] = []
        matched_keys = set()

        # Generate candidate phrases from longest to shortest (tri-grams, bi-grams, unigrams)
        phrases = []
        # Tri-grams
        for i in range(len(words) - 2):
            phrases.append((" ".join(words[i : i + 3]), 3))
        # Bi-grams
        for i in range(len(words) - 1):
            phrases.append((" ".join(words[i : i + 2]), 2))
        # Unigrams (and plural/singular variations)
        for w in words:
            phrases.append((w, 1))
            if w.endswith("s") and len(w) > 3:
                phrases.append((w[:-1], 1))
            if w.endswith("es") and len(w) > 4:
                phrases.append((w[:-2], 1))

        # Check for direct keyword matches
        for phrase, weight in phrases:
            if phrase in self.keyword_index and phrase not in matched_keys:
                item = self.keyword_index[phrase]
                matched_items.append(item)
                matched_keys.add(phrase)
                if len(matched_items) >= top_k:
                    break

        # Check for category matches if still need more matches
        if len(matched_items) < top_k:
            for w in words:
                if w in self.by_category and w not in matched_keys:
                    cat_items = self.by_category[w]
                    if cat_items:
                        matched_items.append(cat_items[0])
                        matched_keys.add(w)
                    if len(matched_items) >= top_k:
                        break

        # Substring / partial match fallback if no exact hits found
        if not matched_items:
            for kw, item in self.keyword_index.items():
                if any(w in kw for w in words if len(w) > 3):
                    if item.name not in matched_keys:
                        matched_items.append(item)
                        matched_keys.add(item.name)
                        if len(matched_items) >= top_k:
                            break

        return matched_items[:top_k]

    def format_conditioning_context(self, examples: List[DatasetItem]) -> str:
        """
        Formats a structured textual reference describing the dataset examples.
        This is injected into the model prompt to guide generation style & subject details.
        """
        if not examples:
            return ""

        descriptions = []
        for ex in examples:
            descriptions.append(
                f"- Concept '{ex.name}' (Category: {ex.category}): authentic visual representation, "
                f"clear subject isolation, natural lighting, high fidelity."
            )

        context = (
            "Visual Reference Style Guidelines (derived from dataset examples):\n"
            + "\n".join(descriptions)
            + "\nEnsure the generated subject exhibits clear silhouette, realistic textures, and vibrant balanced colors matching these references."
        )
        return context


# Global singleton instance
_DATASET_INSTANCE: Optional[ImageDataset] = None


def get_dataset() -> ImageDataset:
    global _DATASET_INSTANCE
    if _DATASET_INSTANCE is None:
        _DATASET_INSTANCE = ImageDataset()
    return _DATASET_INSTANCE


if __name__ == "__main__":
    ds = get_dataset()
    print(f"Dataset successfully indexed {ds.total_images()} images across {len(ds.get_categories())} categories.")
    print("Categories:", ", ".join(ds.get_categories()))

    test_prompts = [
        "a playful dog running in green grass with a ball",
        "a smiling doctor holding a notebook",
        "a fresh mango and banana on a kitchen table",
        "a red car under a rainbow during rain",
    ]

    print("\n--- Testing Prompt Concept Matching ---")
    for tp in test_prompts:
        matches = ds.match_prompt(tp)
        print(f"\nPrompt: '{tp}'")
        print(f"Matched Examples: {[m.name + ' (' + m.category + ')' for m in matches]}")
        print("Paths:", [m.rel_path for m in matches])


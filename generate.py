"""
Command-Line Interface for Dataset-Conditioned Text-to-Image Generation
========================================================================

Usage:
    python generate.py --prompt "a smiling doctor holding a notebook"
    python generate.py --prompt "a red sports car under rain" --provider openai
    python generate.py --prompt "fresh mango and coconut on a table" --provider gemini --output mango.jpg
    python generate.py --check-keys
    python generate.py --browse-dataset
"""

import argparse
import sys
from pathlib import Path
from PIL import Image

from api_key import get_status_info, get_api_key, save_api_key, PROVIDER_NAMES
from dataset_loader import get_dataset
from model_generator import get_model, OUTPUTS_DIR


def print_banner():
    print("=" * 75)
    print("      DATASET-CONDITIONED TEXT-TO-IMAGE GENERATION MODEL")
    print("=" * 75)


def check_keys():
    print_banner()
    info = get_status_info()
    print("\nAPI Key Configuration Status:")
    print(f"  * Default Provider: {info['active_provider'].upper()}")
    print(f"  * Any Key Configured: {'YES' if info['has_any_key'] else 'NO (Pending)'}")
    print("\nProviders:")
    for p_id, p_info in info["providers"].items():
        tag = "[READY]  " if p_info["has_key"] else "[MISSING]"
        print(f"  {tag} {p_info['name']:<35} : {p_info['preview']}")

    print("\nHow to add your API key:")
    print("  1. Quickest: Paste your API key directly into 'api_key.txt'")
    print("  2. Or set it in '.env' (e.g. GEMINI_API_KEY=..., OPENAI_API_KEY=...)")
    print("  3. Or run: python generate.py --set-key <YOUR_KEY> --provider <gemini|openai|stability|huggingface>")
    print("=" * 75)


def browse_dataset():
    print_banner()
    ds = get_dataset()
    cats = ds.get_categories()
    print(f"\nLocal Image Dataset Summary:")
    print(f"  Total Concept Images: {ds.total_images()}")
    print(f"  Total Categories:     {len(cats)}\n")
    for cat in cats:
        items = ds.get_items_by_category(cat)
        names = ", ".join([it.name for it in items[:8]])
        extra = f" (+{len(items)-8} more)" if len(items) > 8 else ""
        print(f"  - {cat.title():<15} ({len(items):>2} images): {names}{extra}")
    print("=" * 75)


def main():
    parser = argparse.ArgumentParser(
        description="Dataset-Conditioned Text-to-Image Generation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--prompt",
        "-p",
        type=str,
        help="Text prompt describing the image you want to generate",
    )
    parser.add_argument(
        "--provider",
        type=str,
        choices=["gemini", "openai", "stability", "huggingface"],
        default=None,
        help="Model provider to use (default: gemini or auto-detected from key)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Output filename (saved in outputs/ directory)",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Do not save output image to disk",
    )
    parser.add_argument(
        "--check-keys",
        action="store_true",
        help="Inspect current API key configuration across all providers",
    )
    parser.add_argument(
        "--set-key",
        type=str,
        default=None,
        help="Quickly save an API key to api_key.txt and .env",
    )
    parser.add_argument(
        "--browse-dataset",
        action="store_true",
        help="List all dataset categories and sample concept images",
    )

    args = parser.parse_args()

    # Handle informational flags
    if args.check_keys:
        check_keys()
        return

    if args.browse_dataset:
        browse_dataset()
        return

    if args.set_key:
        p = args.provider or "gemini"
        save_api_key(args.set_key, provider=p)
        print(f"[Success] Saved API key for {p.upper()} to api_key.txt and .env")
        return

    # If no prompt is provided, show interactive prompt or help
    prompt = args.prompt
    if not prompt:
        print_banner()
        print("\nNo prompt specified via --prompt.")
        try:
            prompt = input("Enter your image generation prompt: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting.")
            sys.exit(0)

        if not prompt:
            print("Prompt cannot be empty. Use --help for usage instructions.")
            sys.exit(1)

    print_banner()
    print(f"\nTarget Prompt: '{prompt}'")

    # Match dataset examples
    ds = get_dataset()
    matches = ds.match_prompt(prompt, top_k=4)

    print(f"\n[1/3] Dataset Example Retrieval:")
    if matches:
        print(f"  Found {len(matches)} reference concept(s) in local dataset:")
        for idx, m in enumerate(matches, 1):
            print(f"   {idx}. '{m.name}' [{m.category}] -> {m.rel_path}")
    else:
        print("  No direct keyword matched in local dataset. Using general generative style.")

    # Initialize model
    model = get_model(provider=args.provider)
    active_p, key = model.get_effective_provider_and_key(args.provider)

    print(f"\n[2/3] Model & Provider Configuration:")
    print(f"  * Provider: {PROVIDER_NAMES.get(active_p, active_p)}")
    print(f"  * API Key Status: {'Configured' if key else 'Pending (Using Preview Mode)'}")

    # Generate
    print(f"\n[3/3] Generating image...")
    res = model.generate(
        prompt=prompt,
        provider=args.provider,
        save_output=not args.no_save,
        output_filename=args.output,
    )

    print("\n" + "=" * 75)
    print("                         RESULT")
    print("=" * 75)
    if res["success"]:
        print("Status: SUCCESS")
        if res["saved_path"]:
            print(f"Saved Image: {res['saved_path']}")
        print(f"\nConditioned Prompt Used by Model:\n{res['conditioned_prompt']}")
        if res["error"]:
            print(f"\nImportant Notice:\n{res['error']}")
    else:
        print(f"Status: FAILED")
        print(f"Error: {res['error']}")
    print("=" * 75)


if __name__ == "__main__":
    main()


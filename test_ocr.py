#!/usr/bin/env python3
"""Test multiple OCR models on dictionary page images and compare results.

Supports: Gemini 2.5 Pro, GPT-4o, Claude Sonnet, Surya OCR.
API keys are read from environment variables or a .env file.

Usage:
    # Test all available models on a single image
    python test_ocr.py test_pages/roerich_page_0050.png

    # Test specific models
    python test_ocr.py test_pages/*.png --models gemini gpt4o

    # Test on a directory of images
    python test_ocr.py test_pages/ --models gemini claude surya

    # Compare existing outputs
    python test_ocr.py test_pages/roerich_page_0050.png --compare-only

Environment variables (or .env file):
    GOOGLE_API_KEY    - for Gemini
    OPENAI_API_KEY    - for GPT-4o
    ANTHROPIC_API_KEY - for Claude
"""

import argparse
import base64
import glob
import os
import sys
import time
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


OCR_PROMPT = """OCR this dictionary page. Extract ALL text exactly as printed, preserving:
- The original layout and line structure
- All scripts: Tibetan (བོད་ཡིག), English, Russian (Русский), Sanskrit/Devanagari (संस्कृत)
- Entry numbering, indentation, and formatting
- Diacritical marks and special characters

This is a page from an old Tibetan-English-Russian-Sanskrit dictionary.
Output the raw text only, no commentary."""

STRUCTURED_PROMPT = """OCR this dictionary page and extract structured entries.
This is from an old Tibetan-English-Russian-Sanskrit dictionary.

For each dictionary entry on the page, output:
- headword (in Tibetan script)
- wylie (Wylie transliteration if you can determine it)
- english (English definition/translation)
- russian (Russian translation, if present)
- sanskrit (Sanskrit/Devanagari equivalent, if present)

Output as a list of entries. Preserve all diacritical marks."""


def encode_image(image_path):
    """Read and base64-encode an image file."""
    with open(image_path, "rb") as f:
        return base64.standard_b64encode(f.read()).decode("utf-8")


def get_mime_type(image_path):
    ext = Path(image_path).suffix.lower()
    return {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".webp": "image/webp",
        ".tiff": "image/tiff",
        ".tif": "image/tiff",
    }.get(ext, "image/png")


# -- Gemini --

def ocr_gemini(image_path, structured=False):
    """OCR using Google Gemini 2.5 Pro."""
    import google.generativeai as genai

    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        return None, "GOOGLE_API_KEY not set"

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-2.5-pro-preview-06-05")

    with open(image_path, "rb") as f:
        image_data = f.read()

    prompt = STRUCTURED_PROMPT if structured else OCR_PROMPT
    mime = get_mime_type(image_path)

    start = time.time()
    response = model.generate_content([
        prompt,
        {"mime_type": mime, "data": image_data},
    ])
    elapsed = time.time() - start

    return response.text, f"{elapsed:.1f}s"


# -- GPT-4o --

def ocr_gpt4o(image_path, structured=False):
    """OCR using OpenAI GPT-4o."""
    from openai import OpenAI

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return None, "OPENAI_API_KEY not set"

    client = OpenAI(api_key=api_key)
    b64 = encode_image(image_path)
    mime = get_mime_type(image_path)
    prompt = STRUCTURED_PROMPT if structured else OCR_PROMPT

    start = time.time()
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{mime};base64,{b64}",
                            "detail": "high",
                        },
                    },
                ],
            }
        ],
        max_tokens=4096,
    )
    elapsed = time.time() - start

    return response.choices[0].message.content, f"{elapsed:.1f}s"


# -- Claude --

def ocr_claude(image_path, structured=False):
    """OCR using Anthropic Claude Sonnet."""
    import anthropic

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return None, "ANTHROPIC_API_KEY not set"

    client = anthropic.Anthropic(api_key=api_key)
    b64 = encode_image(image_path)
    mime = get_mime_type(image_path)
    prompt = STRUCTURED_PROMPT if structured else OCR_PROMPT

    start = time.time()
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=4096,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": mime,
                            "data": b64,
                        },
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ],
    )
    elapsed = time.time() - start

    return response.content[0].text, f"{elapsed:.1f}s"


# -- Surya --

def ocr_surya(image_path, structured=False):
    """OCR using Surya (local, open-source)."""
    try:
        from surya.recognition import RecognitionPredictor
        from surya.detection import DetectionPredictor
        from PIL import Image
    except ImportError:
        return None, "surya-ocr not installed (pip install surya-ocr)"

    start = time.time()

    det_predictor = DetectionPredictor()
    rec_predictor = RecognitionPredictor()

    image = Image.open(image_path)
    predictions = rec_predictor([image], [["bo", "en", "ru", "sa"]], det_predictor)

    lines = []
    for page in predictions:
        for line in page.text_lines:
            lines.append(line.text)

    elapsed = time.time() - start
    return "\n".join(lines), f"{elapsed:.1f}s"


# -- Dispatcher --

MODEL_FUNCS = {
    "gemini": ("Gemini 2.5 Pro", ocr_gemini),
    "gpt4o": ("GPT-4o", ocr_gpt4o),
    "claude": ("Claude Sonnet", ocr_claude),
    "surya": ("Surya OCR", ocr_surya),
}


def save_result(image_path, model_name, text, output_dir):
    """Save OCR result to a text file alongside the image."""
    os.makedirs(output_dir, exist_ok=True)
    stem = Path(image_path).stem
    safe_model = model_name.lower().replace(" ", "_").replace(".", "")
    output_path = os.path.join(output_dir, f"{stem}_{safe_model}.txt")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(text)
    return output_path


def run_comparison(image_paths, models, output_dir, structured=False):
    """Run OCR comparison across models and images."""
    results = {}

    for image_path in image_paths:
        print(f"\n{'='*60}")
        print(f"Image: {image_path}")
        print(f"{'='*60}")
        results[image_path] = {}

        for model_key in models:
            if model_key not in MODEL_FUNCS:
                print(f"  Unknown model: {model_key}")
                continue

            model_name, func = MODEL_FUNCS[model_key]
            print(f"\n  [{model_name}] Processing...", end=" ", flush=True)

            try:
                text, timing = func(image_path, structured=structured)
                if text is None:
                    print(f"SKIPPED ({timing})")
                    continue
                print(f"OK ({timing})")

                out_path = save_result(image_path, model_name, text, output_dir)
                print(f"    Saved: {out_path}")
                print(f"    Characters: {len(text)}")
                preview = text[:200].replace("\n", " ")
                print(f"    Preview: {preview}...")

                results[image_path][model_key] = {
                    "text": text,
                    "timing": timing,
                    "output_path": out_path,
                    "chars": len(text),
                }
            except Exception as e:
                print(f"ERROR: {e}")
                results[image_path][model_key] = {"error": str(e)}

    return results


def print_summary(results):
    """Print a comparison summary table."""
    print(f"\n\n{'='*60}")
    print("COMPARISON SUMMARY")
    print(f"{'='*60}")

    for image_path, model_results in results.items():
        print(f"\n{Path(image_path).name}:")
        for model_key, data in model_results.items():
            model_name = MODEL_FUNCS[model_key][0]
            if "error" in data:
                print(f"  {model_name:20s} ERROR: {data['error']}")
            else:
                print(f"  {model_name:20s} {data['chars']:6d} chars  {data['timing']}")


def main():
    parser = argparse.ArgumentParser(
        description="Compare OCR models on dictionary page images"
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        help="Image file(s) or directory containing images",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=list(MODEL_FUNCS.keys()),
        choices=list(MODEL_FUNCS.keys()),
        help="Models to test (default: all available)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./ocr_results",
        help="Directory to save results (default: ./ocr_results)",
    )
    parser.add_argument(
        "--structured",
        action="store_true",
        help="Use structured extraction prompt instead of raw OCR",
    )
    parser.add_argument(
        "--compare-only",
        action="store_true",
        help="Only show comparison of existing results",
    )

    args = parser.parse_args()

    # Gather image paths
    image_paths = []
    for inp in args.inputs:
        if os.path.isdir(inp):
            for ext in ("*.png", "*.jpg", "*.jpeg", "*.tiff", "*.tif"):
                image_paths.extend(sorted(glob.glob(os.path.join(inp, ext))))
        elif os.path.isfile(inp):
            image_paths.append(inp)
        else:
            expanded = glob.glob(inp)
            image_paths.extend(sorted(expanded))

    if not image_paths:
        print("Error: no image files found")
        sys.exit(1)

    print(f"Found {len(image_paths)} image(s)")
    print(f"Models: {', '.join(MODEL_FUNCS[m][0] for m in args.models)}")
    print(f"Output: {args.output_dir}")

    results = run_comparison(
        image_paths, args.models, args.output_dir, structured=args.structured
    )
    print_summary(results)


if __name__ == "__main__":
    main()

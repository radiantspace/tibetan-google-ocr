# Tibetan Dictionary OCR

Multi-model OCR pipeline for digitizing old Tibetan-English-Russian-Sanskrit dictionaries
(Roerich, Chandra Das, Jaschke).

## Quick Start

```bash
# Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Copy and fill in API keys
cp .env.example .env

# Extract test pages from a dictionary PDF
python extract_pages.py ~/dictionaries/roerich.pdf --pages 50,100,200

# Run OCR comparison across all models
python test_ocr.py test_pages/ --models gemini gpt4o claude surya

# Run with structured extraction (headword, definitions, translations)
python test_ocr.py test_pages/ --models gemini --structured
```

## Tools

### `extract_pages.py` - PDF to images
Extracts specific pages from dictionary PDFs as high-res PNG images.

```bash
python extract_pages.py <pdf_path> [--pages 1,50,100] [--dpi 300] [--output-dir ./test_pages]
python extract_pages.py <pdf_path> --info  # just show page count
```

### `test_ocr.py` - Multi-model OCR comparison
Tests multiple OCR models on the same images and saves results side-by-side.

Supported models:
- **gemini** - Google Gemini 2.5 Pro (needs `GOOGLE_API_KEY`)
- **gpt4o** - OpenAI GPT-4o (needs `OPENAI_API_KEY`)
- **claude** - Anthropic Claude Sonnet (needs `ANTHROPIC_API_KEY`)
- **surya** - Surya OCR, local open-source (no API key needed, `pip install surya-ocr`)

Results are saved to `./ocr_results/` as text files named `{page}_{model}.txt`.

### `ocr.py` - Legacy Google Drive OCR
Original OCR tool using Google Drive API with Wylie transliteration. See file header for setup.

## Supported Scripts

These dictionaries contain 4 scripts on the same pages:
- **Tibetan** (Uchen script)
- **English** (Latin)
- **Russian** (Cyrillic)
- **Sanskrit** (Devanagari)

## API Keys

| Model | Env Variable | Get a key |
|-------|-------------|-----------|
| Gemini | `GOOGLE_API_KEY` | https://aistudio.google.com/apikey |
| GPT-4o | `OPENAI_API_KEY` | https://platform.openai.com/api-keys |
| Claude | `ANTHROPIC_API_KEY` | https://console.anthropic.com/ |

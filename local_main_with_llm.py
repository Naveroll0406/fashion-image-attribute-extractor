# local_main.py
"""
Integrated pipeline (Ollama-only for Mistral):
 - BLIP (local) -> image captioning
 - Ollama (local) -> Mistral attribute extraction via REST API
 - outputs -> results.csv, results.json, results_with_attrs.xlsx, results_skipped.json

Pre-req:
 - Ollama running locally and a mistral model pulled: `ollama pull mistral` (or let the script pull it)
 - Ollama REST API default: http://localhost:11434

This file is an updated version with reliability and logging improvements:
 - BLIP: explicit amp.autocast usage; fp16 fallback preserved
 - Ollama: retry mechanism, model readiness checks, auto-pull option, clearer logging
 - Image validation/fetch: more robust content-type / magic-bytes checks
 - JSON parsing: improved balanced-brace extraction
 - Pipeline: more informative logging per image, normalized URL matching for Excel output
 - Minor fixes: consistent sleep/backoff, retries, progress logging
"""

import os
import sys
import time
import json
import re
import imghdr
from io import BytesIO
from typing import Optional, Dict
from dotenv import load_dotenv
import requests
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

import torch
from transformers import BlipForConditionalGeneration, BlipProcessor

# -----------------------------
# CONFIG - edit paths / prefs
# -----------------------------
EXCEL_PATH = r"C:\Users\naver\Desktop\Fashion\Best_Seller_Tags.xlsx"
SHEET_NAME = "Tagging"
COL_NAME = "Image URL"

OUTPUT_CSV = "results.csv"
OUTPUT_JSON = "results.json"
OUTPUT_XL = "results_with_attrs.xlsx"
SKIPPED_JSON = "results_skipped.json"

MAX_LINKS = 100              # set None to process all
ENV_PATH = r"C:\Users\naver\Desktop\Fashion\.env"
REQUEST_TIMEOUT = 15
SLEEP_BETWEEN = 0.3
CREATE_PLOTS = False
VERBOSE = True
REQUEST_RETRIES = 3
BACKOFF_FACTOR = 1.5

# BLIP model (base is lighter for GTX 1650)
LOCAL_BLIP_MODEL = "Salesforce/blip-image-captioning-base"
BLIP_GENERATE_KWARGS = {"max_length": 64, "num_beams": 5, "early_stopping": True}

# Ollama / Mistral settings
# Use explicit Ollama model identifier. Ollama model names follow the `model:tag` format.
# Examples: "mistral:latest", "mistral:instruct", "mistral:7b-instruct-v0.3-q5_K_M"
OLLAMA_API_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL_NAME = "mistral:7b-instruct-q4_K_M"   # default -> change if you prefer a different tag
OLLAMA_TIMEOUT = 120
OLLAMA_PULL_TIMEOUT = 3600  # max seconds to wait while pulling/loading a model
OLLAMA_POLL_INTERVAL = 5
AUTO_PULL_MODEL = True       # attempt to pull model if not found

# LLM generation reliability settings
LLM_MAX_TOKENS = 256
LLM_TEMPERATURE = 0.0

# -----------------------------
# Taxonomy / regex keyword maps
# (same as your earlier mappings) - preserved
# -----------------------------
NECKLINE_KEYWORDS = {
    "Sweetheart": [r"sweetheart", r"heart[- ]?shaped"],
    "Plunging": [r"plunging", r"deep plunge", r"deep v-neck", r"deep v"],
    "V-neck": [r"\bv[- ]?neck\b", r"\bv neck\b", r"\bv-neck\b"],
    "Crew": [r"\bcrew\b", r"round neck", r"round neckline", r"crewneck", r"high crew"],
    "Halter": [r"halter", r"halterneck", r"halter neck"],
    "Square": [r"square neck", r"square neckline", r"\bsquare\b neck"],
    "Off-shoulder": [r"off[- ]?shoulder", r"off the shoulder", r"bardot"],
    "Boat": [r"boat neck", r"bateau", r"boat neckline", r"bateau neck"],
    "Illusion": [r"illusion neckline", r"\billusion\b"],
    "Scoop": [r"scoop neck", r"scoop neckline"],
    "Cowl": [r"cowl neck", r"cowl neckline"],
    "Asymmetric": [r"asymmetr", r"one-shoulder", r"one shoulder"],
    "High neck": [r"high neck", r"mock neck", r"turtleneck", r"stand[- ]?up collar"],
}

SILHOUETTE_KEYWORDS = {
    "A-line": [r"\ba[- ]?line\b", r"\ba line\b", r"a-line skirt"],
    "Mermaid": [r"mermaid", r"fishtail", r"trumpet"],
    "Sheath": [r"sheath", r"column", r"bodycon", r"body[- ]?con", r"figure[- ]?hugging"],
    "Ball gown": [r"ball gown", r"ballgown", r"princess", r"full skirt"],
    "Fit and flare": [r"fit[- ]?and[- ]?flare", r"fit and flare", r"fit[- ]?&[- ]?flare"],
    "Empire": [r"\bempire\b", r"empire waist", r"underbust"],
    "Slip": [r"slip dress", r"bias cut", r"bias[- ]?cut"],
    "Shift": [r"\bshift dress\b", r"\bshift\b"],
    "Peplum": [r"peplum"],
    "Skater": [r"skater dress", r"flared skirt", r"skater"],
    "Tea length": [r"tea[- ]?length", r"tea length"],
    "Trapeze": [r"trapeze"],
    "Wrap": [r"wrap dress", r"wrap[- ]?style"],
    "Kaftan": [r"kaftan", r"caftan"],
    "Slip/Bias": [r"bias cut", r"bias[- ]?cut"],
}

WAISTLINE_KEYWORDS = {
    "Empire": [r"empire waist", r"\bempire\b", r"underbust"],
    "Natural": [r"natural waist", r"\bnatural\b", r"at the waist", r"waistline at natural"],
    "Drop waist": [r"drop waist", r"dropped waist", r"low waist"],
    "High waist": [r"high waist", r"high[- ]?waist"],
    "Basque": [r"basque waist", r"basque"],
    "Fitted waist": [r"fitted waist", r"cinched waist", r"cinched"],
    "Corset": [r"corset", r"corseted", r"lace[- ]?up bodice"],
}

SLEEVES_KEYWORDS = {
    "Sleeveless": [r"\bsleeveless\b", r"\bstrapless\b", r"tank style", r"spaghetti strap", r"spaghetti[- ]?strap"],
    "Cap sleeves": [r"cap sleeve", r"cap sleeves"],
    "Short sleeves": [r"\bshort sleeve\b", r"short[- ]?sleeve", r"short sleeves"],
    "3/4 sleeves": [r"3/?4", r"three[- ]?quarter", r"three quarter"],
    "Long sleeves": [r"\blong sleeve\b", r"long[- ]?sleeve", r"full sleeve", r"full[- ]?length sleeves"],
    "Puff sleeves": [r"puff sleeve", r"puffy sleeve", r"puff sleeves", r"puffed sleeve"],
    "Bell sleeves": [r"bell sleeve", r"bell sleeves", r"flared sleeve"],
    "Bishop sleeves": [r"bishop sleeve", r"bishop sleeves"],
    "Flutter sleeves": [r"flutter sleeve", r"flutter sleeves"],
    "Kimono sleeves": [r"kimono sleeve", r"kimono sleeves"],
    "Cold shoulder": [r"cold shoulder", r"cold-shoulder"],
    "Raglan": [r"raglan sleeve", r"raglan sleeves"],
    "Lantern": [r"lantern sleeve", r"lantern sleeves"],
}


def _compile_map(keyword_map):
    compiled = {}
    for label, pats in keyword_map.items():
        compiled[label] = [re.compile(p, re.IGNORECASE) for p in pats]
    return compiled

COMPILED_NECKLINE = _compile_map(NECKLINE_KEYWORDS)
COMPILED_SILHOUETTE = _compile_map(SILHOUETTE_KEYWORDS)
COMPILED_WAISTLINE = _compile_map(WAISTLINE_KEYWORDS)
COMPILED_SLEEVES = _compile_map(SLEEVES_KEYWORDS)

# -----------------------------
# BLIP loader & caption
# -----------------------------
def load_local_blip_model(model_name=LOCAL_BLIP_MODEL):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[BLIP] Loading {model_name} -> device: {device}")
    # set use_fast=True to silence processor warning and get faster tokenizer
    processor = BlipProcessor.from_pretrained(model_name, use_fast=True)
    # attempt fp16 on CUDA to save VRAM; fall back to CPU if fails
    try:
        if device == "cuda":
            # attempt to load with fp16
            model = BlipForConditionalGeneration.from_pretrained(model_name, torch_dtype=torch.float16)
            model.to("cuda")
        else:
            model = BlipForConditionalGeneration.from_pretrained(model_name)
            model.to("cpu")
    except Exception as e:
        print("[BLIP] fp16 cuda load failed or not supported; falling back to CPU load:", e)
        model = BlipForConditionalGeneration.from_pretrained(model_name)
        model.to("cpu")
    model.eval()
    return processor, model, device


def caption_from_image_bytes(image_bytes, processor, model, device, generate_kwargs=None):
    if generate_kwargs is None:
        generate_kwargs = {}
    image = Image.open(BytesIO(image_bytes)).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")
    if device == "cuda":
        inputs = {k: v.to("cuda") for k, v in inputs.items()}
        # use torch.amp.autocast with device string to avoid deprecated keyword usage
        try:
            with torch.amp.autocast("cuda", dtype=torch.float16):
                generated_ids = model.generate(**inputs, **generate_kwargs)
        except Exception as e:
            # if autocast fails for any reason, fallback to regular generation
            print("[BLIP] autocast generation failed, falling back to normal fp32 generation:", e)
            generated_ids = model.generate(**inputs, **generate_kwargs)
    else:
        inputs = {k: v.to("cpu") for k, v in inputs.items()}
        generated_ids = model.generate(**inputs, **generate_kwargs)
    caption = processor.tokenizer.decode(generated_ids[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)
    return caption.strip()

# -----------------------------
# Regex fallback extractor (original)
# -----------------------------
def _match_first(compiled_map, text):
    for label, patterns in compiled_map.items():
        for pat in patterns:
            if pat.search(text):
                return label
    return None


def extract_attributes_from_caption(caption):
    txt = caption.lower()
    neckline = _match_first(COMPILED_NECKLINE, txt)
    silhouette = _match_first(COMPILED_SILHOUETTE, txt)
    waistline = _match_first(COMPILED_WAISTLINE, txt)
    sleeves = _match_first(COMPILED_SLEEVES, txt)
    if silhouette is None and re.search(r"fitted|tight|bodycon|figure[- ]?hugging|slim fit", txt):
        silhouette = "Sheath"
    if sleeves is None and re.search(r"strap|spaghetti", txt):
        sleeves = "Sleeveless"
    return {
        "neckline": neckline or "Unknown",
        "silhouette": silhouette or "Unknown",
        "waistline": waistline or "Unknown",
        "sleeves": sleeves or "Unknown",
        "source_caption": caption,
    }

# -----------------------------
# Ollama (always used) generation
# -----------------------------
ATTR_PROMPT = """You are a fashion expert. Given the short description of a garment, extract these four fields as JSON only:
- "neckline": (e.g., V-neck, Sweetheart, Crew, Halter, Square, Off-shoulder, Scoop, etc.)
- "silhouette": (e.g., A-line, Mermaid, Sheath, Ball gown, Fit and flare, Empire, Slip, Shift, Peplum, Skater, etc.)
- "waistline": (e.g., Empire, Natural, Drop waist, High waist, Basque, Fitted waist, Corset)
- "sleeves": (e.g., Sleeveless, Cap sleeves, Short sleeves, 3/4 sleeves, Long sleeves, Puff sleeves, Bell sleeves, etc.)

Description: "{caption}"

Return ONLY a JSON object with these four keys. Use 'Unknown' if you cannot determine a field.
"""


def _normalize_model_name(m: str) -> str:
    if not m or not isinstance(m, str):
        return ""
    m = m.strip()
    # if user provided a short name like 'mistral', default the tag to :latest
    if ":" not in m:
        m = f"{m}:latest"
    return m


def _ensure_model_ready(model_name: str, timeout: int = 30) -> bool:
    """Attempt a lightweight check to ensure Ollama has the model available. If missing, optionally trigger a pull.
    Returns True if model appears available/loaded or load initiated successfully.
    """
    model_name = _normalize_model_name(model_name)
    if VERBOSE:
        print(f"[Ollama] ensuring model ready: {model_name}")
    try:
        # ask Ollama to load the model into memory by sending an empty prompt
        resp = requests.post(OLLAMA_API_URL, json={"model": model_name}, timeout=timeout)
        if resp.status_code == 200:
            # successful call; server either loaded it or confirmed presence
            try:
                j = resp.json()
                # if we got a response object, treat as success
                if VERBOSE:
                    print(f"[Ollama] model check response keys: {list(j.keys())}")
                return True
            except Exception:
                return True
        else:
            # 4xx could mean model not found
            text = resp.text if 'resp' in locals() else ''
            if VERBOSE:
                print(f"[Ollama] model check returned status {resp.status_code}: {text}")
            if resp.status_code in (404, 400) and AUTO_PULL_MODEL:
                # try to pull model
                print(f"[Ollama] model {model_name} not present locally. Attempting to pull via /api/pull...")
                try:
                    p = requests.post(OLLAMA_API_URL.replace('/api/generate', '/api/pull'), json={"model": model_name}, timeout=60)
                    if p.status_code == 200:
                        # poll until loaded or timeout
                        start = time.time()
                        print("[Ollama] pulling model; this may take a while (check ollama logs). Waiting until model is available...")
                        while time.time() - start < OLLAMA_PULL_TIMEOUT:
                            try:
                                r2 = requests.post(OLLAMA_API_URL, json={"model": model_name}, timeout=30)
                                if r2.status_code == 200:
                                    print("[Ollama] model appears loaded.")
                                    return True
                            except Exception:
                                pass
                            time.sleep(OLLAMA_POLL_INTERVAL)
                        print(f"[Ollama] timed out waiting for model pull ({OLLAMA_PULL_TIMEOUT}s).")
                        return False
                    else:
                        print(f"[Ollama] pull failed: status {p.status_code} - {p.text}")
                        return False
                except Exception as e:
                    print(f"[Ollama] pull request failed: {e}")
                    return False
            return False
    except requests.exceptions.RequestException as e:
        print(f"[Ollama] connection error during model check: {e}")
        return False


def generate_with_ollama(prompt: str, model_name: str = OLLAMA_MODEL_NAME, max_tokens=LLM_MAX_TOKENS, temperature=LLM_TEMPERATURE, retries=REQUEST_RETRIES):
    model_name = _normalize_model_name(model_name)
    # quick model readiness check (handles first-time model load)
    ready = _ensure_model_ready(model_name, timeout=20)
    if not ready:
        raise RuntimeError(f"Ollama model {model_name} not available or failed to load. Ensure Ollama is running and model is pulled.")

    body = {
        "model": model_name,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": False
    }
    attempt = 0
    while attempt < retries:
        try:
            if VERBOSE:
                print(f"[Ollama] Request attempt {attempt+1}/{retries} (timeout={OLLAMA_TIMEOUT}s) -> model={model_name}")
            resp = requests.post(OLLAMA_API_URL, json=body, timeout=OLLAMA_TIMEOUT)
            resp.raise_for_status()
            # try parsing possible response shapes
            try:
                data = resp.json()
            except Exception:
                return resp.text
            if isinstance(data, dict):
                if "result" in data and isinstance(data["result"], str):
                    return data["result"]
                if "text" in data and isinstance(data["text"], str):
                    return data["text"]
                if "outputs" in data and isinstance(data["outputs"], list) and len(data["outputs"])>0:
                    out = data["outputs"][0]
                    if isinstance(out, dict) and "content" in out:
                        return out["content"]
                    return str(out)
                # new-style 'response' key
                if "response" in data:
                    return data["response"]
            if isinstance(data, list) and data:
                return str(data[0])
            return json.dumps(data)
        except requests.exceptions.RequestException as e:
            wait = BACKOFF_FACTOR ** attempt
            print(f"[Ollama] request failed (attempt {attempt+1}): {e} - retrying in {wait:.1f}s")
            time.sleep(wait)
            attempt += 1
    raise RuntimeError(f"Ollama request failed after {retries} attempts")


def _try_parse_json_from_text(text: str) -> Optional[Dict]:
    if not text or not isinstance(text, str):
        return None
    start = text.find('{')
    if start == -1:
        try:
            return json.loads(text)
        except Exception:
            return None
    depth = 0
    end = -1
    for i in range(start, len(text)):
        ch = text[i]
        if ch == '{':
            depth += 1
        elif ch == '}':
            depth -= 1
            if depth == 0:
                end = i
                break
    if end == -1:
        try:
            return json.loads(text)
        except Exception:
            try:
                return json.loads(text.replace("'", '"'))
            except Exception:
                return None
    cand = text[start:end+1]
    try:
        return json.loads(cand)
    except Exception:
        try:
            return json.loads(cand.replace("'", '"'))
        except Exception:
            return None


def extract_with_ollama(caption: str) -> Dict:
    prompt = ATTR_PROMPT.format(caption=caption)
    try:
        text = generate_with_ollama(prompt, model_name=OLLAMA_MODEL_NAME, max_tokens=LLM_MAX_TOKENS, temperature=LLM_TEMPERATURE)
        obj = _try_parse_json_from_text(text)
        if obj:
            def v(k):
                val = obj.get(k, None)
                if not val or (isinstance(val, str) and val.strip()==""):
                    return "Unknown"
                return str(val).strip()
            res = {
                "neckline": v("neckline"),
                "silhouette": v("silhouette"),
                "waistline": v("waistline"),
                "sleeves": v("sleeves"),
                "source_caption": caption
            }
            if all(res[k]=="Unknown" for k in ("neckline","silhouette","waistline","sleeves")):
                return extract_attributes_from_caption(caption)
            return res
        else:
            print("[Ollama] Could not parse JSON from model output; falling back to regex.")
            if VERBOSE:
                print("[Ollama] raw output:", text)
            return extract_attributes_from_caption(caption)
    except Exception as e:
        print("[Ollama] generation error:", e)
        return extract_attributes_from_caption(caption)

# -----------------------------
# Networking helpers
# -----------------------------

def validate_image_url(url):
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        r = requests.head(url, allow_redirects=True, timeout=REQUEST_TIMEOUT, headers=headers)
        if r.status_code >= 400:
            return False, f"HEAD status {r.status_code}"
        ctype = r.headers.get("content-type", "") or ""
        if "image" in ctype.lower():
            return True, None
        r2 = requests.get(url, stream=True, timeout=REQUEST_TIMEOUT, headers=headers)
        if r2.status_code != 200:
            return False, f"GET status {r2.status_code}"
        first_bytes = r2.raw.read(2048) if hasattr(r2, 'raw') else r2.content[:2048]
        if imghdr.what(None, first_bytes) is not None:
            return True, None
        try:
            Image.open(BytesIO(first_bytes)).verify()
            return True, None
        except Exception:
            return False, "not an image or unknown content-type"
    except Exception as e:
        return False, str(e)


def fetch_image_bytes(url, retries=REQUEST_RETRIES):
    headers = {"User-Agent": "Mozilla/5.0"}
    attempt = 0
    while attempt < retries:
        try:
            resp = requests.get(url, timeout=REQUEST_TIMEOUT, headers=headers)
            resp.raise_for_status()
            return resp.content
        except Exception as e:
            wait = BACKOFF_FACTOR ** attempt
            print(f"[Network] failed to download image (attempt {attempt+1}): {e} - retrying in {wait:.1f}s")
            time.sleep(wait)
            attempt += 1
    raise RuntimeError(f"Failed to download image after {retries} attempts: {url}")

# -----------------------------
# Main pipeline
# -----------------------------

def normalize_url(u: str) -> str:
    if not isinstance(u, str):
        return ""
    return u.strip().rstrip('/')


def process_from_excel():
    if os.path.exists(ENV_PATH):
        load_dotenv(dotenv_path=ENV_PATH)
        print(f"Loaded .env from: {ENV_PATH}")
    else:
        load_dotenv()
        print("No explicit .env, loaded environment fallback.")

    # normalize model name early
    global OLLAMA_MODEL_NAME
    OLLAMA_MODEL_NAME = _normalize_model_name(OLLAMA_MODEL_NAME)
    if VERBOSE:
        print(f"[Ollama] using model: {OLLAMA_MODEL_NAME}")

    # Load BLIP
    processor, blip_model, blip_device = load_local_blip_model()

    # Read Excel
    if not os.path.exists(EXCEL_PATH):
        print(f"ERROR: Excel file not found at: {EXCEL_PATH}")
        sys.exit(1)
    try:
        df = pd.read_excel(EXCEL_PATH, sheet_name=SHEET_NAME)
    except Exception as e:
        print("ERROR reading Excel:", e)
        sys.exit(1)
    if COL_NAME not in df.columns:
        print(f"ERROR: Column '{COL_NAME}' not found in sheet '{SHEET_NAME}'")
        print("Available columns:", list(df.columns))
        sys.exit(1)

    urls_raw = df[COL_NAME].dropna().astype(str).tolist()
    total = len(urls_raw)
    if MAX_LINKS is not None:
        urls_raw = urls_raw[:MAX_LINKS]
    print(f"Processing {len(urls_raw)} of {total} URLs from {EXCEL_PATH} -> {SHEET_NAME} -> {COL_NAME}")

    results = []
    skipped = []
    attrs_by_norm_url = {}

    for i, url in enumerate(tqdm(urls_raw, desc="Processing images", unit="img")):
        url = url.strip()
        if not url:
            continue
        if VERBOSE:
            print(f"\n[{i+1}/{len(urls_raw)}] Processing: {url}")
        ok, reason = validate_image_url(url)
        if not ok:
            skipped.append({"url": url, "reason": reason})
            if VERBOSE:
                print(f"Skipped: {reason}")
            continue
        try:
            img_bytes = fetch_image_bytes(url)
        except Exception as e:
            skipped.append({"url": url, "reason": f"download error: {e}"})
            if VERBOSE:
                print(f"Download error: {e}")
            time.sleep(SLEEP_BETWEEN)
            continue

        # caption via BLIP
        try:
            caption = caption_from_image_bytes(img_bytes, processor, blip_model, blip_device, generate_kwargs=BLIP_GENERATE_KWARGS)
            if VERBOSE:
                print(f"Caption: {caption}")
        except Exception as e:
            skipped.append({"url": url, "reason": f"blip error: {e}"})
            if VERBOSE:
                print(f"BLIP error: {e}")
            time.sleep(SLEEP_BETWEEN)
            continue

        # attributes via Ollama (Mistral)
        try:
            attrs = extract_with_ollama(caption)
            if VERBOSE:
                print(f"Extracted attrs: {attrs}")
        except Exception as e:
            print("Attribute extraction error:", e)
            attrs = extract_attributes_from_caption(caption)

        record = {
            "image_url": url,
            "caption": attrs.get("source_caption", caption),
            "neckline": attrs.get("neckline", "Unknown"),
            "silhouette": attrs.get("silhouette", "Unknown"),
            "waistline": attrs.get("waistline", "Unknown"),
            "sleeves": attrs.get("sleeves", "Unknown"),
        }
        results.append(record)
        attrs_by_norm_url[normalize_url(url)] = record
        time.sleep(SLEEP_BETWEEN)

    # save results
    if results:
        out_df = pd.DataFrame(results)
        out_df.to_csv(OUTPUT_CSV, index=False)
        print(f"Saved {len(out_df)} rows to {OUTPUT_CSV}")
        with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"Saved JSON to {OUTPUT_JSON}")

        # append attributes to original Excel rows by matching normalized URLs (safe mapping)
        try:
            df_out = df.copy()
            # new columns with defaults (but do not clobber existing columns if present)
            if "caption" not in df_out.columns:
                df_out["caption"] = ""
            else:
                df_out["caption"] = df_out["caption"].astype(str)
            for col in ["neckline", "silhouette", "waistline", "sleeves"]:
                if col not in df_out.columns:
                    df_out[col] = "Unknown"
                else:
                    df_out[col] = df_out[col].astype(str)

            # fill for rows where URL matches (using normalized URL keys)
            for idx, row in df_out.iterrows():
                url = str(row.get(COL_NAME, "") or "").strip()
                nurl = normalize_url(url)
                if nurl and nurl in attrs_by_norm_url:
                    rec = attrs_by_norm_url[nurl]
                    df_out.at[idx, "caption"] = rec["caption"]
                    df_out.at[idx, "neckline"] = rec["neckline"]
                    df_out.at[idx, "silhouette"] = rec["silhouette"]
                    df_out.at[idx, "waistline"] = rec["waistline"]
                    df_out.at[idx, "sleeves"] = rec["sleeves"]
            df_out.to_excel(OUTPUT_XL, index=False)
            print(f"Saved Excel with attributes to {OUTPUT_XL}")
        except Exception as e:
            print("Could not save excel:", e)

        if CREATE_PLOTS:
            try:
                plot_attribute_distributions(out_df)
            except Exception as e:
                print("Could not create plots:", e)
    else:
        print("No results to save.")

    if skipped:
        with open(SKIPPED_JSON, "w", encoding="utf-8") as f:
            json.dump(skipped, f, indent=2, ensure_ascii=False)
        print(f"Saved skipped list ({len(skipped)}) to {SKIPPED_JSON}")


def plot_attribute_distributions(df):
    for col in ["neckline", "silhouette", "sleeves"]:
        counts = df[col].value_counts()
        plt.figure(figsize=(8,4))
        counts.plot(kind="bar")
        plt.title(f"Distribution of {col}")
        plt.ylabel("count")
        plt.tight_layout()
        out = f"distribution_{col}.png"
        plt.savefig(out)
        plt.close()
        print(f"Saved plot: {out}")

if __name__ == "__main__":
    print("Starting integrated pipeline (BLIP -> Ollama Mistral).")
    process_from_excel()
    print("Done.")

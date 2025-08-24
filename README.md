# Fashion Attribute Extraction Pipeline (BLIP + Ollama Mistral)

This project integrates **BLIP** (for image captioning) with **Ollama Mistral** (for attribute extraction) to automatically extract **fashion attributes** such as neckline, silhouette, waistline, and sleeves from product images.
![Yoru](https://github.com/user-attachments/assets/3b67b230-4ec9-4a6e-bd4a-fba5757fb5d0)

---

## 🚀 Features
- **BLIP (local)**: Generates captions for product images.
- **Mistral via Ollama (local)**: Extracts structured attributes from captions.
- **Excel/CSV/JSON output**: Saves results in multiple formats for easy analysis.
- **Image validation**: Skips invalid/broken image URLs.
- **Retry & logging**: More robust and reliable pipeline.
- **Plots (optional)**: Distribution of extracted attributes.

---

## 📂 Input
The script reads product image URLs from an Excel file:

- **File:** `Best_Seller_Tags.xlsx`  
- **Sheet:** `Tagging`  
- **Column:** `Image URL`

---

## 📦 Installation

1. Clone this repo (or copy the files).
2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

   ⚠️ **PyTorch installation** depends on your system:
   - **CUDA 12.1 (GPU):**
     ```bash
     pip install torch==2.3.1+cu121 --index-url https://download.pytorch.org/whl/cu121
     ```
   - **CPU-only:**
     ```bash
     pip install torch==2.3.1+cpu --index-url https://download.pytorch.org/whl/cpu
     ```

---

## ⚙️ Pre-requisites

- **Ollama** must be installed and running locally → [Install Ollama](https://ollama.ai)
- Pull the **Mistral** model:
  ```bash
  ollama pull mistral:7b-instruct-q4_K_M
  ```
- Place your Excel input file at the path configured in the script (`EXCEL_PATH`).

---

## ▶️ Running the Pipeline

```bash
python local_main_with_llm.py
```

### Outputs
- `results.csv` → CSV with image URL, caption, and attributes
- `results.json` → JSON list of extracted records
- `results_with_attrs.xlsx` → Excel with attributes appended
- `results_skipped.json` → Skipped/broken image URLs
---

## 🧩 Example Output

```json
{
  "image_url": "https://example.com/dress.jpg",
  "caption": "a woman wearing a purple dress with a thigh slit",
  "neckline": "V-neck",
  "silhouette": "Sheath",
  "waistline": "Natural",
  "sleeves": "Sleeveless"
}
```

---

## ⚡ Notes
- Adjust `EXCEL_PATH`, `SHEET_NAME`, and `COL_NAME` in the script if your input differs.
- `MAX_LINKS` can be set to limit processing for testing.
- If Ollama model is not found, the script will attempt to auto-pull it.

---

## 📜 License
This project is for **demo/interview purposes**. Modify as needed.

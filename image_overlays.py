import json
from pathlib import Path

import fitz  # PyMuPDF: pip install pymupdf
import matplotlib.pyplot as plt
from PIL import Image

JSON_FILE = (
    "./data/parsed/DCO/2023/"
    "Evaluation of the United Nations Philippines Cooperation Framework "
    "2019-2023_331/"
    "Evaluation of the United Nations Philippines Cooperation Framework "
    "2019-2023_331.json"
)
PDF_FILE = (
    "./data/parsed/DCO/2023/"
    "Evaluation of the United Nations Philippines Cooperation Framework "
    "2019-2023_331/"
    "Evaluation of the United Nations Philippines Cooperation Framework "
    "2019-2023_331.pdf"
)

PDF_PATH = Path(PDF_FILE)
JSON_PATH = Path(JSON_FILE)

# 1. Load Docling JSON
with open(JSON_PATH, "r", encoding="utf-8") as f:
    docling = json.load(f)

# Docling structure:
# - docling["pages"] is a dict with string keys: "1", "2", "3", ...
# - docling["texts"] is a flat list of all text items
# - each text item has "prov" (provenance) with page_no and bbox

pages_dict = docling["pages"]
all_texts = docling["texts"]

# 2. Open PDF
pdf = fitz.open(str(PDF_PATH))

# 3. Process each page
for page_idx, page in enumerate(pdf):
    page_no = page_idx + 1  # PDF pages are 1-indexed in the JSON

    # 3. Render the page to an image
    pix = page.get_pixmap()
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

    fig, ax = plt.subplots(figsize=(10, 13))
    ax.imshow(img)
    ax.set_axis_off()

    page_height = img.height  # needed for Y flipping

    # 4. Draw bounding boxes for texts on this page
    for text_item in all_texts:
        # Each text can have multiple provenance entries (if split across pages)
        for prov in text_item.get("prov", []):
            if prov.get("page_no") != page_no:
                continue

            bbox = prov["bbox"]  # dict with l, t, r, b keys
            # l=left, t=top, r=right, b=bottom
            # coord_origin is BOTTOMLEFT (PDF coordinates)
            x0, y0, x1, y1 = bbox["l"], bbox["b"], bbox["r"], bbox["t"]

            # Convert bottom-left origin -> top-left origin for the image
            # PDF y increases upwards; image y increases downwards
            y0_img = page_height - y1
            height = y1 - y0
            width = x1 - x0

            rect = plt.Rectangle(
                (x0, y0_img), width, height, fill=False, edgecolor="red", linewidth=0.5
            )
            ax.add_patch(rect)

    plt.tight_layout()
    plt.show()

    # Optional: only process first page for testing
    # break

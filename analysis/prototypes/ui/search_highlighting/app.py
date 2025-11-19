"""
Proof-of-Concept: PDF Viewer with Dynamic Text Highlighting
Uses Docling JSON to highlight search results in PDFs
"""

import json
from pathlib import Path

from flask import Flask, jsonify, render_template, request, send_file

app = Flask(__name__)

# Configuration
BASE_DIR = Path(__file__).parent.parent.parent.parent.parent
PDF_DIR = BASE_DIR / "data" / "parsed" / "DCO" / "2023"
SAMPLE_DOC = (
    "Evaluation of the United Nations Philippines Cooperation Framework 2019-2023_331"
)


@app.route("/")
def index():
    """Landing page with sample search results"""
    return render_template(
        "index.html",
        documents=[
            {
                "id": "sample_331",
                "title": "UN Philippines Cooperation Framework 2019-2023",
                "excerpt": "Evaluation of the United Nations...",
            }
        ],
    )


@app.route("/viewer/<doc_id>")
def viewer(doc_id):
    """PDF viewer page"""
    return render_template("viewer.html", doc_id=doc_id)


@app.route("/api/pdf/<doc_id>")
def serve_pdf(doc_id):
    """Serve the PDF file"""
    if doc_id == "sample_331":
        pdf_path = PDF_DIR / SAMPLE_DOC / f"{SAMPLE_DOC}.pdf"
        if pdf_path.exists():
            return send_file(pdf_path, mimetype="application/pdf")
    return jsonify({"error": "PDF not found"}), 404


@app.route("/api/highlight/<doc_id>")
def get_highlight_data(doc_id):
    """
    Get bounding box data for highlighting text blocks
    Query params:
    - page: page number to highlight on
    - text: text to search for (optional, returns all texts if not specified)
    """
    if doc_id == "sample_331":
        json_path = PDF_DIR / SAMPLE_DOC / f"{SAMPLE_DOC}.json"

        if not json_path.exists():
            return jsonify({"error": "JSON not found"}), 404

        with open(json_path, "r", encoding="utf-8") as f:
            docling_data = json.load(f)

        page_no = request.args.get("page", type=int)
        search_text = request.args.get("text", "").lower()

        # Find matching texts
        highlights = []

        for text_item in docling_data.get("texts", []):
            text_content = text_item.get("text", "")

            # Filter by search text if provided
            if search_text and search_text not in text_content.lower():
                continue

            # Get bounding boxes
            for prov in text_item.get("prov", []):
                if page_no and prov.get("page_no") != page_no:
                    continue

                bbox = prov.get("bbox", {})
                highlights.append(
                    {
                        "page": prov.get("page_no"),
                        "bbox": bbox,
                        "text": text_content,
                        "coord_origin": bbox.get("coord_origin", "BOTTOMLEFT"),
                    }
                )

        return jsonify({"highlights": highlights, "total": len(highlights)})

    return jsonify({"error": "Document not found"}), 404


@app.route("/api/search")
def search():
    """
    Search endpoint - searches through Docling JSON for matching text
    Returns: list of results with page and bbox info
    """
    query = request.args.get("q", "").strip()

    if not query:
        return jsonify({"results": []})

    # Search through the actual Docling JSON
    json_path = PDF_DIR / SAMPLE_DOC / f"{SAMPLE_DOC}.json"

    if not json_path.exists():
        return jsonify({"results": []})

    with open(json_path, "r", encoding="utf-8") as f:
        docling_data = json.load(f)

    results = []
    seen_pages = set()  # Track pages we've already added to avoid duplicates

    # Search through all text items
    for text_item in docling_data.get("texts", []):
        text_content = text_item.get("text", "")

        # Case-insensitive search
        if query.lower() in text_content.lower():
            # Get the page number from provenance
            for prov in text_item.get("prov", []):
                page_no = prov.get("page_no")

                # Only add one result per page to avoid duplicates
                if page_no and page_no not in seen_pages:
                    seen_pages.add(page_no)

                    # Create a snippet with context
                    snippet = text_content[:100] + (
                        "..." if len(text_content) > 100 else ""
                    )

                    results.append(
                        {
                            "doc_id": "sample_331",
                            "title": "UN Philippines Cooperation Framework 2019-2023",
                            "page": page_no,
                            "text": query,  # The search term
                            "snippet": snippet,
                        }
                    )

                    # Limit results for performance
                    if len(results) >= 10:
                        break

        if len(results) >= 10:
            break

    # Sort by page number
    results.sort(key=lambda x: x["page"])

    return jsonify({"results": results})


if __name__ == "__main__":
    print("Starting PDF Highlight POC...")
    print(f"Sample PDF path: {PDF_DIR / SAMPLE_DOC}")
    app.run(debug=True, port=5000)

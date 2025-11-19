# PDF Highlight Proof-of-Concept

Web application that demonstrates dynamic PDF highlighting based on search results using Docling JSON data.

## Features

- 🔍 **Search Interface**: Mock search page showing potential results
- 📄 **PDF Viewer**: Full-featured PDF viewer using PDF.js
- 🎯 **Dynamic Highlights**: Overlay highlights on PDF text using bounding box data from Docling
- ✨ **Interactive**: Users can zoom, navigate pages, select and copy text
- 🚀 **Real-time**: Highlights are generated on-the-fly from JSON data

## Architecture

### Backend (Flask)
- Serves PDFs from the parsed data directory
- Provides API endpoints for highlight data extraction from Docling JSON
- Mock search endpoint (in production, this would query your actual search index)

### Frontend (HTML + PDF.js)
- PDF.js for rendering PDFs in the browser
- Custom highlight overlay system using absolute-positioned divs
- Coordinate transformation from PDF space to canvas space

## Setup

1. Install dependencies:
```bash
cd ui/test
pip install -r requirements.txt
```

2. Run the server:
```bash
python app.py
```

3. Open in browser:
```
http://localhost:5000
```

## How It Works

1. **User clicks a search result** with metadata: `{doc_id, page, text}`
2. **Backend loads** the corresponding Docling JSON file
3. **Backend filters** text items matching the query
4. **Frontend receives** bbox coordinates for matching text
5. **Frontend overlays** semi-transparent highlights on the PDF canvas
6. **User can** zoom, pan, copy text - the PDF remains fully interactive

## API Endpoints

- `GET /` - Search interface
- `GET /viewer/<doc_id>` - PDF viewer page
- `GET /api/pdf/<doc_id>` - Serve PDF file
- `GET /api/highlight/<doc_id>?page=X&text=Y` - Get highlight bounding boxes
- `GET /api/search?q=query` - Search documents (mock)

## Demo Links

Once running, try these:
- http://localhost:5000/viewer/sample_331?page=1&text=evaluation
- http://localhost:5000/viewer/sample_331?page=5&text=recommendations
- http://localhost:5000/viewer/sample_331?page=1 (show all text bboxes)

## Production Considerations

For production deployment, you would need to:

1. **Search Backend**: Replace mock search with actual search engine (Elasticsearch, Meilisearch, etc.)
2. **Caching**: Cache highlight data to reduce JSON parsing overhead
3. **Authentication**: Add user authentication and access control
4. **Optimization**:
   - Lazy load highlights per page
   - Compress large JSON files
   - Use CDN for static assets
5. **Scaling**: Use proper WSGI server (Gunicorn, uWSGI) instead of Flask dev server

## Technology Stack

- **Backend**: Flask (Python)
- **Frontend**: Vanilla JavaScript + PDF.js
- **Data Source**: Docling JSON output
- **PDF Rendering**: Mozilla PDF.js library

## Notes

- Currently hardcoded to work with the sample Philippines document
- In production, you'd need to handle multiple documents and dynamic paths
- The highlight overlay approach works well but for very dense text, consider using PDF.js text layer

import React, { useState, useEffect, useRef } from 'react';
import axios from 'axios';
import API_BASE_URL from '../config';
import { HighlightBox } from '../types/api';

// PDF.js types
declare global {
  interface Window {
    pdfjsLib: any;
  }
}

interface PDFViewerProps {
  docId: string;
  chunkId: string;
  pageNum?: number;
  onClose: () => void;
  title?: string;
}

export const PDFViewer: React.FC<PDFViewerProps> = ({
  docId,
  chunkId,
  pageNum = 1,
  onClose,
  title = 'Document'
}) => {
  const [pdfDoc, setPdfDoc] = useState<any>(null);
  const [currentPage, setCurrentPage] = useState(pageNum);
  const [totalPages, setTotalPages] = useState(0);
  const [scale, setScale] = useState(1.5);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [highlights, setHighlights] = useState<HighlightBox[]>([]);
  const [highlightCount, setHighlightCount] = useState(0);

  const containerRef = useRef<HTMLDivElement>(null);
  const pageContainerRef = useRef<HTMLDivElement>(null);

  // Update current page when pageNum prop changes
  useEffect(() => {
    console.log('PDFViewer: pageNum changed to', pageNum);
    setCurrentPage(pageNum);
  }, [pageNum]);

  // Load PDF when docId changes
  useEffect(() => {
    console.log('PDFViewer: docId changed to', docId);
    const initPDF = async () => {
      if (window.pdfjsLib) {
        // Already loaded, just load the PDF
        await loadPDF();
        return;
      }

      // Load PDF.js library first time only
      const script = document.createElement('script');
      script.src = 'https://cdnjs.cloudflare.com/ajax/libs/pdf.js/3.11.174/pdf.min.js';
      script.async = true;
      script.onload = async () => {
        window.pdfjsLib.GlobalWorkerOptions.workerSrc =
          'https://cdnjs.cloudflare.com/ajax/libs/pdf.js/3.11.174/pdf.worker.min.js';
        await loadPDF();
      };
      script.onerror = () => {
        setError('Failed to load PDF.js library');
        setLoading(false);
      };
      document.body.appendChild(script);
    };

    initPDF();
  }, [docId]);

  // Load highlight data when chunkId changes
  useEffect(() => {
    console.log('PDFViewer: loading highlights for chunk', chunkId);
    loadHighlights();
  }, [chunkId]);

  // Re-render page when page changes or scale changes
  useEffect(() => {
    if (pdfDoc) {
      renderPage(currentPage);
    }
  }, [currentPage, scale, pdfDoc, highlights]);

  const loadPDF = async () => {
    setLoading(true);
    setError(null);

    try {
      console.log('Loading PDF:', docId, 'page:', pageNum);
      const url = `${API_BASE_URL}/pdf/${docId}`;
      const loadingTask = window.pdfjsLib.getDocument(url);
      const pdf = await loadingTask.promise;

      setPdfDoc(pdf);
      setTotalPages(pdf.numPages);
      setLoading(false);
      console.log('PDF loaded, total pages:', pdf.numPages);
    } catch (err: any) {
      console.error('Error loading PDF:', err);
      setError(`Failed to load PDF: ${err.message}`);
      setLoading(false);
    }
  };

  const loadHighlights = async () => {
    try {
      // Get highlights for the specific chunk
      const response = await axios.get<{ highlights: HighlightBox[]; total: number }>(
        `${API_BASE_URL}/highlight/chunk/${chunkId}`
      );

      setHighlights(response.data.highlights || []);
      console.log(`Loaded ${response.data.total} highlight positions for chunk ${chunkId}`);
    } catch (err) {
      console.error('Error loading highlights:', err);
      setHighlights([]);
    }
  };

  const renderPage = async (pageNumber: number) => {
    if (!pdfDoc || !pageContainerRef.current) return;

    try {
      const page = await pdfDoc.getPage(pageNumber);
      const viewport = page.getViewport({ scale });

      // Clear container
      pageContainerRef.current.innerHTML = '';
      pageContainerRef.current.style.width = `${viewport.width}px`;
      pageContainerRef.current.style.height = `${viewport.height}px`;

      // Create canvas
      const canvas = document.createElement('canvas');
      const context = canvas.getContext('2d');
      if (!context) return;

      // High DPI support
      const outputScale = window.devicePixelRatio || 1;
      canvas.width = Math.floor(viewport.width * outputScale);
      canvas.height = Math.floor(viewport.height * outputScale);
      canvas.style.width = `${Math.floor(viewport.width)}px`;
      canvas.style.height = `${Math.floor(viewport.height)}px`;
      canvas.style.display = 'block';

      // Scale context for retina
      const transform = outputScale !== 1
        ? [outputScale, 0, 0, outputScale, 0, 0]
        : null;

      pageContainerRef.current.appendChild(canvas);

      // Render PDF page
      await page.render({
        canvasContext: context,
        viewport: viewport,
        transform: transform
      }).promise;

      // Add text layer for selectability
      const textLayerDiv = document.createElement('div');
      textLayerDiv.className = 'textLayer';
      textLayerDiv.style.width = `${viewport.width}px`;
      textLayerDiv.style.height = `${viewport.height}px`;
      textLayerDiv.style.position = 'absolute';
      textLayerDiv.style.left = '0';
      textLayerDiv.style.top = '0';
      textLayerDiv.style.overflow = 'hidden';
      textLayerDiv.style.opacity = '1';
      textLayerDiv.style.lineHeight = '1.0';

      pageContainerRef.current.appendChild(textLayerDiv);

      const textContent = await page.getTextContent();
      window.pdfjsLib.renderTextLayer({
        textContent: textContent,
        container: textLayerDiv,
        viewport: viewport,
        textDivs: []
      });

      // Add highlights
      addHighlights(pageNumber, viewport);

    } catch (err) {
      console.error('Error rendering page:', err);
    }
  };

  const addHighlights = (pageNumber: number, viewport: any) => {
    if (!pageContainerRef.current) return;

    const pageHighlights = highlights.filter(h => h.page === pageNumber);
    setHighlightCount(pageHighlights.length);

    const margin = 2;

    pageHighlights.forEach(highlight => {
      const bbox = highlight.bbox;

      // Convert PDF coordinates (bottom-left origin) to canvas coordinates (top-left origin)
      const x = (bbox.l - margin) * scale;
      const y = (viewport.height / scale - bbox.t - margin) * scale;
      const width = ((bbox.r - bbox.l) + (margin * 2)) * scale;
      const height = ((bbox.t - bbox.b) + (margin * 2)) * scale;

      // Create highlight div
      const highlightDiv = document.createElement('div');
      highlightDiv.className = 'highlight-overlay';
      highlightDiv.style.position = 'absolute';
      highlightDiv.style.pointerEvents = 'none';
      highlightDiv.style.background = 'rgba(255, 235, 59, 0.4)';
      highlightDiv.style.border = '2px solid rgba(255, 193, 7, 0.8)';
      highlightDiv.style.boxSizing = 'border-box';
      highlightDiv.style.zIndex = '1';
      highlightDiv.style.left = `${x}px`;
      highlightDiv.style.top = `${y}px`;
      highlightDiv.style.width = `${width}px`;
      highlightDiv.style.height = `${height}px`;
      highlightDiv.title = highlight.text;

      pageContainerRef.current!.appendChild(highlightDiv);
    });
  };

  const goToPrevPage = () => {
    if (currentPage > 1) {
      setCurrentPage(currentPage - 1);
    }
  };

  const goToNextPage = () => {
    if (currentPage < totalPages) {
      setCurrentPage(currentPage + 1);
    }
  };

  const zoomIn = () => {
    setScale(scale + 0.25);
  };

  const zoomOut = () => {
    if (scale > 0.5) {
      setScale(scale - 0.25);
    }
  };

  const handlePageInput = (e: React.ChangeEvent<HTMLInputElement>) => {
    const pageNum = parseInt(e.target.value);
    if (pageNum >= 1 && pageNum <= totalPages) {
      setCurrentPage(pageNum);
    }
  };

  if (loading) {
    return (
      <div className="pdf-viewer-container">
        <div className="pdf-viewer-loading">Loading PDF...</div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="pdf-viewer-container">
        <div className="pdf-viewer-error">
          <p>{error}</p>
          <button onClick={onClose}>Close</button>
        </div>
      </div>
    );
  }

  return (
    <div className="pdf-viewer-container">
      <div className="pdf-viewer-header">
        <div className="pdf-viewer-title-row">
          <h4 title={title}>{title}</h4>
          <button onClick={onClose} className="close-button" title="Close">✕</button>
        </div>
        <div className="pdf-viewer-controls">
          <button onClick={goToPrevPage} disabled={currentPage <= 1}>
            Previous
          </button>
          <span className="page-info">
            Page{' '}
            <input
              type="number"
              min="1"
              max={totalPages}
              value={currentPage}
              onChange={handlePageInput}
              className="page-input"
            />
            {' '}of {totalPages}
          </span>
          <button onClick={goToNextPage} disabled={currentPage >= totalPages}>
            Next
          </button>
        </div>
      </div>

      <div className="pdf-viewer-content" ref={containerRef}>
        <div
          ref={pageContainerRef}
          className="pdf-page-container"
        />
      </div>
    </div>
  );
};

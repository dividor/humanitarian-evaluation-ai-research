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
  const [scale] = useState(1.5);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [highlights, setHighlights] = useState<HighlightBox[]>([]);
  const [renderedPages, setRenderedPages] = useState<Set<number>>(new Set());

  const containerRef = useRef<HTMLDivElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const textLayerRef = useRef<HTMLDivElement>(null);
  const highlightsRef = useRef<HTMLDivElement>(null);
  const renderTaskRef = useRef<any>(null);

  // Load PDF
  useEffect(() => {
    const initPDF = async () => {
      if (window.pdfjsLib) {
        await loadPDF();
        return;
      }

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

  // Load highlights
  useEffect(() => {
    if (chunkId && pdfDoc) {
      loadHighlights();
    }
  }, [chunkId, pdfDoc]);

  // Render current page when it changes
  useEffect(() => {
    if (pdfDoc && currentPage > 0) {
      renderPage(currentPage);
    }
  }, [pdfDoc, currentPage, highlights]);

  const loadPDF = async () => {
    try {
      const url = `${API_BASE_URL}/pdf/${docId}`;
      const loadingTask = window.pdfjsLib.getDocument(url);
      const pdf = await loadingTask.promise;

      setPdfDoc(pdf);
      setTotalPages(pdf.numPages);
      setCurrentPage(pageNum);
      setLoading(false);
    } catch (err: any) {
      setError(`Failed to load PDF: ${err.message}`);
      setLoading(false);
    }
  };

  const loadHighlights = async () => {
    try {
      const response = await axios.get<{ highlights: HighlightBox[]; total: number }>(
        `${API_BASE_URL}/highlight/chunk/${chunkId}`
      );
      setHighlights(response.data.highlights || []);
    } catch (err) {
      console.error('Error loading highlights:', err);
    }
  };

  const renderPage = async (pageNumber: number) => {
    if (!pdfDoc || !canvasRef.current || !textLayerRef.current || !highlightsRef.current) return;

    try {
      // Cancel any ongoing render
      if (renderTaskRef.current) {
        renderTaskRef.current.cancel();
      }

      const page = await pdfDoc.getPage(pageNumber);
      const viewport = page.getViewport({ scale });

      const canvas = canvasRef.current;
      const context = canvas.getContext('2d');
      if (!context) return;

      // High DPI
      const outputScale = window.devicePixelRatio || 1;
      canvas.width = Math.floor(viewport.width * outputScale);
      canvas.height = Math.floor(viewport.height * outputScale);
      canvas.style.width = `${viewport.width}px`;
      canvas.style.height = `${viewport.height}px`;

      // Render PDF
      const renderContext = {
        canvasContext: context,
        viewport: viewport,
        transform: outputScale !== 1 ? [outputScale, 0, 0, outputScale, 0, 0] : null
      };

      renderTaskRef.current = page.render(renderContext);
      await renderTaskRef.current.promise;
      renderTaskRef.current = null;

      // Text layer
      const textLayer = textLayerRef.current;
      textLayer.innerHTML = '';
      textLayer.style.width = `${viewport.width}px`;
      textLayer.style.height = `${viewport.height}px`;
      textLayer.style.setProperty('--scale-factor', scale.toString());

      const textContent = await page.getTextContent();
      window.pdfjsLib.renderTextLayer({
        textContent,
        container: textLayer,
        viewport,
        textDivs: []
      });

      // Highlights
      const highlightsContainer = highlightsRef.current;
      highlightsContainer.innerHTML = '';

      const pageHighlights = highlights.filter(h => h.page === pageNumber);
      console.log(`Rendering ${pageHighlights.length} highlights for page ${pageNumber}`);

      pageHighlights.forEach(highlight => {
        const { bbox } = highlight;
        const x = bbox.l * scale;
        const y = (viewport.height / scale - bbox.t) * scale;
        const width = (bbox.r - bbox.l) * scale;
        const height = (bbox.t - bbox.b) * scale;

        const div = document.createElement('div');
        div.className = 'highlight-overlay';
        Object.assign(div.style, {
          position: 'absolute',
          pointerEvents: 'none',
          background: 'rgba(255, 235, 59, 0.4)',
          border: '2px solid rgba(255, 193, 7, 0.8)',
          zIndex: '10',
          left: `${x}px`,
          top: `${y}px`,
          width: `${width}px`,
          height: `${height}px`
        });
        div.title = highlight.text.substring(0, 100);
        highlightsContainer.appendChild(div);
      });

      setRenderedPages(prev => new Set([...prev, pageNumber]));
    } catch (err: any) {
      if (err.name === 'RenderingCancelledException') {
        console.log('Rendering cancelled for page', pageNumber);
      } else {
        console.error(`Error rendering page ${pageNumber}:`, err);
      }
    }
  };

  const goToPage = (page: number) => {
    if (page >= 1 && page <= totalPages) {
      setCurrentPage(page);
    }
  };

  const handleWheel = (e: React.WheelEvent) => {
    if (!containerRef.current) return;

    const container = containerRef.current;
    const atTop = container.scrollTop === 0;
    const atBottom = container.scrollTop + container.clientHeight >= container.scrollHeight - 5;

    // Scroll to next page if at bottom
    if (e.deltaY > 0 && atBottom && currentPage < totalPages) {
      e.preventDefault();
      goToPage(currentPage + 1);
      setTimeout(() => {
        if (containerRef.current) containerRef.current.scrollTop = 0;
      }, 10);
    }
    // Scroll to previous page if at top
    else if (e.deltaY < 0 && atTop && currentPage > 1) {
      e.preventDefault();
      goToPage(currentPage - 1);
      setTimeout(() => {
        if (containerRef.current) {
          containerRef.current.scrollTop = containerRef.current.scrollHeight;
        }
      }, 10);
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
          <button onClick={onClose} className="close-button">✕</button>
        </div>
        <div className="pdf-viewer-controls">
          <button onClick={() => goToPage(currentPage - 1)} disabled={currentPage <= 1}>
            Previous
          </button>
          <span className="page-info">
            Page{' '}
            <input
              type="number"
              min="1"
              max={totalPages}
              value={currentPage}
              onChange={(e) => {
                const num = parseInt(e.target.value);
                if (!isNaN(num)) goToPage(num);
              }}
              className="page-input"
            />
            {' '}of {totalPages}
          </span>
          <button onClick={() => goToPage(currentPage + 1)} disabled={currentPage >= totalPages}>
            Next
          </button>
        </div>
      </div>

      <div className="pdf-viewer-content" ref={containerRef} onWheel={handleWheel}>
        <div className="pdf-page-container" style={{ position: 'relative' }}>
          <canvas ref={canvasRef} style={{ display: 'block' }} />
          <div
            ref={textLayerRef}
            className="textLayer"
            style={{
              position: 'absolute',
              left: 0,
              top: 0,
              overflow: 'hidden',
              opacity: 1,
              lineHeight: 1.0
            }}
          />
          <div
            ref={highlightsRef}
            style={{ position: 'absolute', left: 0, top: 0, width: '100%', height: '100%' }}
          />
          <div
            style={{
              position: 'absolute',
              top: '10px',
              right: '10px',
              background: 'rgba(0,0,0,0.7)',
              color: 'white',
              padding: '4px 8px',
              borderRadius: '4px',
              fontSize: '12px',
              fontWeight: 'bold',
              pointerEvents: 'none',
              zIndex: 100
            }}
          >
            Page {currentPage}
          </div>
        </div>
      </div>
    </div>
  );
};

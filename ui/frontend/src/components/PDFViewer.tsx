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

const ESTIMATED_PAGE_HEIGHT = 1200; // Approximate height per page for scrollbar
const BUFFER_PAGES = 2; // Number of pages to render before/after current

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
  const [renderedPages, setRenderedPages] = useState<Map<number, boolean>>(new Map());
  const [actualPageHeight, setActualPageHeight] = useState(ESTIMATED_PAGE_HEIGHT);

  const scrollContainerRef = useRef<HTMLDivElement>(null);
  const pagesContainerRef = useRef<HTMLDivElement>(null);
  const renderTasksRef = useRef<Map<number, any>>(new Map());
  const isScrollingProgrammatically = useRef(false);

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

  // Render visible pages when current page changes
  useEffect(() => {
    if (pdfDoc && currentPage > 0) {
      renderVisiblePages();
    }
  }, [pdfDoc, currentPage, highlights, totalPages]);

  // Re-position all rendered pages when actual page height changes
  useEffect(() => {
    if (actualPageHeight !== ESTIMATED_PAGE_HEIGHT && pagesContainerRef.current) {
      // Update positions of all existing page containers
      for (let i = 1; i <= totalPages; i++) {
        const pageContainer = document.getElementById(`pdf-page-${i}`);
        if (pageContainer) {
          pageContainer.style.top = `${(i - 1) * actualPageHeight}px`;
        }
      }
    }
  }, [actualPageHeight, totalPages]);

  // Update scroll position when page changes programmatically
  useEffect(() => {
    if (scrollContainerRef.current && totalPages > 0 && isScrollingProgrammatically.current) {
      const scrollPosition = (currentPage - 1) * actualPageHeight;
      scrollContainerRef.current.scrollTop = scrollPosition;
      isScrollingProgrammatically.current = false;
    }
  }, [currentPage, totalPages, actualPageHeight]);

  // Handle scroll to determine current page
  const handleScroll = () => {
    if (!scrollContainerRef.current || totalPages === 0 || isScrollingProgrammatically.current) return;

    const scrollTop = scrollContainerRef.current.scrollTop;
    const newPage = Math.floor(scrollTop / actualPageHeight) + 1;
    const clampedPage = Math.max(1, Math.min(totalPages, newPage));

    if (clampedPage !== currentPage) {
      setCurrentPage(clampedPage);
    }
  };

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

      // After highlights are loaded, scroll to first one
      if (response.data.highlights && response.data.highlights.length > 0) {
        setTimeout(() => {
          scrollToFirstHighlightWithData(response.data.highlights);
        }, 300);
      }
    } catch (err) {
      console.error('Error loading highlights:', err);
    }
  };

  const scrollToFirstHighlightWithData = (highlightData: HighlightBox[]) => {
    if (!scrollContainerRef.current || highlightData.length === 0) return;

    // Find the first highlight's page
    const firstHighlight = highlightData[0];
    const highlightPage = firstHighlight.page;
    const bbox = firstHighlight.bbox;

    console.log('Scrolling to first highlight:', { page: highlightPage, bbox });

    // Update current page
    setCurrentPage(highlightPage);

    // Scroll to center the highlight in the viewport
    setTimeout(() => {
      if (scrollContainerRef.current) {
        const pageScrollTop = (highlightPage - 1) * actualPageHeight;
        const viewportHeight = scrollContainerRef.current.clientHeight;

        // Calculate highlight position on the page
        // PDF coordinates are from bottom-left, so we need to convert
        // bbox.t is distance from bottom, we want distance from top
        const highlightTopFromBottom = bbox.t * scale;
        const highlightBottomFromBottom = bbox.b * scale;
        const highlightMiddleFromBottom = (highlightTopFromBottom + highlightBottomFromBottom) / 2;

        // Convert to distance from top of page (assuming standard page height ~800-1000)
        // For now, use a rough estimate: middle of highlight as offset from page start
        const highlightMiddleOnPage = actualPageHeight * 0.5; // Rough approximation

        // Center the highlight's middle in the viewport
        const scrollPosition = pageScrollTop + highlightMiddleOnPage - (viewportHeight / 2);

        console.log('Scroll calculation:', {
          pageScrollTop,
          actualPageHeight,
          viewportHeight,
          highlightMiddleOnPage,
          scrollPosition
        });

        scrollContainerRef.current.scrollTop = Math.max(0, scrollPosition);
      }
    }, 100);
  };

  const renderVisiblePages = async () => {
    if (!pdfDoc || !pagesContainerRef.current) return;

    // Calculate range of pages to render (current + buffer)
    const startPage = Math.max(1, currentPage - BUFFER_PAGES);
    const endPage = Math.min(totalPages, currentPage + BUFFER_PAGES);

    // Render pages in range
    for (let pageNum = startPage; pageNum <= endPage; pageNum++) {
      if (!renderedPages.has(pageNum)) {
        await renderPage(pageNum);
      }
    }

    // Clean up pages that are too far from current view
    const pagesToRemove: number[] = [];
    renderedPages.forEach((_, pageNum) => {
      if (pageNum < startPage - BUFFER_PAGES || pageNum > endPage + BUFFER_PAGES) {
        pagesToRemove.push(pageNum);
      }
    });

    if (pagesToRemove.length > 0) {
      const newRenderedPages = new Map(renderedPages);
      pagesToRemove.forEach(pageNum => {
        const pageEl = document.getElementById(`pdf-page-${pageNum}`);
        if (pageEl) {
          pageEl.innerHTML = '';
        }
        newRenderedPages.delete(pageNum);
      });
      setRenderedPages(newRenderedPages);
    }
  };

  const renderPage = async (pageNumber: number) => {
    if (!pdfDoc || !pagesContainerRef.current) return;

    try {
      // Cancel any ongoing render for this page
      const existingTask = renderTasksRef.current.get(pageNumber);
      if (existingTask) {
        existingTask.cancel();
      }

      const page = await pdfDoc.getPage(pageNumber);
      const viewport = page.getViewport({ scale });

      // Update actual page height based on first rendered page
      if (pageNumber === 1 || pageNumber === pageNum) {
        const calculatedHeight = viewport.height + 20; // Add margin
        if (Math.abs(calculatedHeight - actualPageHeight) > 10) {
          setActualPageHeight(calculatedHeight);
        }
      }

      // Get or create page container
      let pageContainer = document.getElementById(`pdf-page-${pageNumber}`) as HTMLDivElement;
      if (!pageContainer) {
        pageContainer = document.createElement('div');
        pageContainer.id = `pdf-page-${pageNumber}`;
        pageContainer.className = 'pdf-page-wrapper';
        pageContainer.style.position = 'absolute';
        pageContainer.style.top = `${(pageNumber - 1) * actualPageHeight}px`;
        pageContainer.style.left = '50%';
        pageContainer.style.transform = 'translateX(-50%)';
        pageContainer.style.overflow = 'visible';
        pageContainer.style.marginBottom = '20px';
        pagesContainerRef.current.appendChild(pageContainer);
      } else {
        // Update position with actual height
        pageContainer.style.top = `${(pageNumber - 1) * actualPageHeight}px`;
      }

      // Clear existing content
      pageContainer.innerHTML = '';

      // Canvas
      const canvas = document.createElement('canvas');
      const context = canvas.getContext('2d');
      if (!context) return;

      const outputScale = window.devicePixelRatio || 1;
      canvas.width = Math.floor(viewport.width * outputScale);
      canvas.height = Math.floor(viewport.height * outputScale);
      canvas.style.width = `${viewport.width}px`;
      canvas.style.height = `${viewport.height}px`;
      canvas.style.display = 'block';

      pageContainer.appendChild(canvas);

      // Render PDF
      const renderContext = {
        canvasContext: context,
        viewport: viewport,
        transform: outputScale !== 1 ? [outputScale, 0, 0, outputScale, 0, 0] : null
      };

      const renderTask = page.render(renderContext);
      renderTasksRef.current.set(pageNumber, renderTask);
      await renderTask.promise;
      renderTasksRef.current.delete(pageNumber);

      // Text layer
      const textLayer = document.createElement('div');
      textLayer.className = 'textLayer';
      Object.assign(textLayer.style, {
        position: 'absolute',
        left: '0',
        top: '0',
        width: `${viewport.width}px`,
        height: `${viewport.height}px`,
        overflow: 'visible',
        opacity: '1',
        lineHeight: '1.0',
        pointerEvents: 'auto'
      });
      textLayer.style.setProperty('--scale-factor', scale.toString());
      pageContainer.appendChild(textLayer);

      const textContent = await page.getTextContent();
      window.pdfjsLib.renderTextLayer({
        textContent,
        container: textLayer,
        viewport,
        textDivs: []
      });

      // Highlights
      const pageHighlights = highlights.filter(h => h.page === pageNumber);
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
        pageContainer.appendChild(div);
      });

      // Page label
      const label = document.createElement('div');
      label.textContent = `Page ${pageNumber}`;
      Object.assign(label.style, {
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
        zIndex: '100'
      });
      pageContainer.appendChild(label);

      // Mark as rendered
      setRenderedPages(prev => new Map(prev).set(pageNumber, true));
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
      isScrollingProgrammatically.current = true;
      setCurrentPage(page);
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

  const totalScrollHeight = totalPages * actualPageHeight;

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

      <div
        className="pdf-viewer-content"
        ref={scrollContainerRef}
        onScroll={handleScroll}
        style={{ overflowY: 'scroll' }}
      >
        <div
          ref={pagesContainerRef}
          style={{ height: `${totalScrollHeight}px`, position: 'relative' }}
        />
      </div>
    </div>
  );
};

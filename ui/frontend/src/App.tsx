import React, { useState, useEffect, useCallback } from 'react';
import axios from 'axios';
import './App.css';
import API_BASE_URL from './config';
import { SearchResponse, Facets, SearchFilters, SearchResult } from './types/api';
import { PDFViewer } from './components/PDFViewer';

function App() {
  const [query, setQuery] = useState('');
  const [results, setResults] = useState<SearchResult[]>([]);
  const [facets, setFacets] = useState<Facets | null>(null);
  const [filters, setFilters] = useState<SearchFilters>({});
  const [loading, setLoading] = useState(false);
  const [selectedDoc, setSelectedDoc] = useState<SearchResult | null>(null);
  const [expandedCards, setExpandedCards] = useState<Set<string>>(new Set());

  // Load facets on mount
  useEffect(() => {
    loadFacets();
  }, []);

  // Debounced search - wait 500ms after user stops typing
  useEffect(() => {
    // Don't search if query is empty
    if (!query.trim()) {
      setResults([]);
      return;
    }

    const timeoutId = setTimeout(() => {
      performSearch();
    }, 500);

    // Cleanup: cancel the timeout if query changes before 500ms
    return () => clearTimeout(timeoutId);
  }, [query, filters]); // Re-run when query or filters change

  const loadFacets = async () => {
    try {
      const response = await axios.get<Facets>(`${API_BASE_URL}/facets`);
      setFacets(response.data);
    } catch (error) {
      console.error('Error loading facets:', error);
    }
  };

  const performSearch = useCallback(async () => {
    if (!query.trim()) return;

    setLoading(true);
    try {
      const params = new URLSearchParams({ q: query, limit: '50' });
      if (filters.organization) params.append('organization', filters.organization);
      if (filters.year) params.append('year', filters.year);
      if (filters.evaluation_type) params.append('evaluation_type', filters.evaluation_type);

      const response = await axios.get<SearchResponse>(`${API_BASE_URL}/search?${params}`);
      setResults(response.data.results);
    } catch (error) {
      console.error('Error searching:', error);
      alert('Search failed. Make sure the backend is running.');
    } finally {
      setLoading(false);
    }
  }, [query, filters]);

  const handleSearch = async (e: React.FormEvent) => {
    e.preventDefault();
    // When form is submitted (Enter key), search immediately
    performSearch();
  };

  const handleFilterChange = (key: keyof SearchFilters, value: string) => {
    setFilters(prev => ({ ...prev, [key]: value }));
  };

  const handleClearFilters = () => {
    setFilters({});
  };

  const handleResultClick = (result: SearchResult) => {
    setSelectedDoc(result);
  };

  const handleClosePreview = () => {
    setSelectedDoc(null);
  };

  const toggleCardExpansion = (chunkId: string, e: React.MouseEvent) => {
    e.stopPropagation(); // Prevent card click from firing
    setExpandedCards(prev => {
      const newSet = new Set(prev);
      if (newSet.has(chunkId)) {
        newSet.delete(chunkId);
      } else {
        newSet.add(chunkId);
      }
      return newSet;
    });
  };

  const hasSearched = results.length > 0 || (query && !loading);

  return (
    <div className="app">
      {/* Top Navigation Bar */}
      <header className="top-bar">
        <div className="top-bar-content">
          <h1 className="app-title">Evidence Lab</h1>
        </div>
      </header>

      {/* Search Box Container */}
      <div className="search-container">
        <form onSubmit={handleSearch} className="search-form">
          <input
            type="text"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Search evaluations..."
            className="search-input"
          />
          <button type="submit" disabled={loading} className="search-button">
            {loading ? 'Searching...' : 'Search'}
          </button>
        </form>
      </div>

      {/* Main Content Area - Only show when results exist */}
      {results.length > 0 && (
        <div className="main-content">
          <div className="content-grid">
            {/* Left Column: Filters */}
            <aside className="filters-section">
              <h2 className="section-heading">Filters</h2>
              <div className="filters-card">
                {facets && (
                  <>
                    <div className="filter-group">
                      <label className="filter-label">Organization</label>
                      <select
                        value={filters.organization || ''}
                        onChange={(e) => handleFilterChange('organization', e.target.value)}
                        className="filter-select"
                      >
                        <option value="">All</option>
                        {facets.organizations.slice(0, 20).map(f => (
                          <option key={f.value} value={f.value}>
                            {f.value} ({f.count})
                          </option>
                        ))}
                      </select>
                    </div>

                    <div className="filter-group">
                      <label className="filter-label">Year</label>
                      <select
                        value={filters.year || ''}
                        onChange={(e) => handleFilterChange('year', e.target.value)}
                        className="filter-select"
                      >
                        <option value="">All</option>
                        {facets.years.slice(0, 10).map(f => (
                          <option key={f.value} value={f.value}>
                            {f.value} ({f.count})
                          </option>
                        ))}
                      </select>
                    </div>

                    <div className="filter-group">
                      <label className="filter-label">Evaluation Type</label>
                      <select
                        value={filters.evaluation_type || ''}
                        onChange={(e) => handleFilterChange('evaluation_type', e.target.value)}
                        className="filter-select"
                      >
                        <option value="">All</option>
                        {facets.evaluation_types.slice(0, 10).map(f => (
                          <option key={f.value} value={f.value}>
                            {f.value} ({f.count})
                          </option>
                        ))}
                      </select>
                    </div>

                    <button
                      onClick={handleClearFilters}
                      className="clear-filters-button"
                    >
                      Clear Filters
                    </button>
                  </>
                )}
              </div>
            </aside>

            {/* Right Column: Search Results */}
            <main className="results-section">
              <h2 className="section-heading">Search Results</h2>

              {/* AI Summary Box */}
              <div className="ai-summary-box">
                <h3 className="ai-summary-title">AI Summary</h3>
                <p className="ai-summary-text">
                  Found {results.length} results for "{query}". The results include evaluations
                  from various organizations covering different themes and geographic regions.
                </p>
              </div>

              {/* Results List */}
              <div className="results-list">
                {results.map((result) => {
                  const isExpanded = expandedCards.has(result.chunk_id);
                  const snippetText = result.text;
                  const displayText = isExpanded ? snippetText : snippetText.substring(0, 200);

                  return (
                    <div
                      key={result.chunk_id}
                      className={`result-card ${selectedDoc?.chunk_id === result.chunk_id ? 'result-card-selected' : ''}`}
                      data-doc-id={result.doc_id}
                      data-page={result.page_num}
                    >
                      <h3
                        className="result-title result-title-link"
                        onClick={() => handleResultClick(result)}
                        role="button"
                        tabIndex={0}
                        onKeyPress={(e) => {
                          if (e.key === 'Enter' || e.key === ' ') {
                            handleResultClick(result);
                          }
                        }}
                      >
                        {result.title}
                      </h3>

                      <div className="result-badges">
                        {result.organization && (
                          <span className="badge badge-org">{result.organization}</span>
                        )}
                        {result.year && (
                          <span className="badge badge-year">{result.year}</span>
                        )}
                        <span className="badge badge-page">Page {result.page_num}</span>
                        <span className="result-score">Score: {result.score.toFixed(3)}</span>
                      </div>

                      {result.headings.length > 0 && (
                        <div className="result-breadcrumb">
                          {result.headings.join(' › ')}
                        </div>
                      )}

                      <p className="result-snippet">
                        {displayText}
                        {!isExpanded && snippetText.length > 200 && (
                          <>
                            ...{' '}
                            <button
                              className="see-more-link"
                              onClick={(e) => toggleCardExpansion(result.chunk_id, e)}
                              aria-label="See more"
                            >
                              Show more
                            </button>
                          </>
                        )}
                        {isExpanded && snippetText.length > 200 && (
                          <>
                            {' '}
                            <button
                              className="see-more-link"
                              onClick={(e) => toggleCardExpansion(result.chunk_id, e)}
                              aria-label="See less"
                            >
                              Show less
                            </button>
                          </>
                        )}
                      </p>
                    </div>
                  );
                })}
              </div>
            </main>
          </div>
        </div>
      )}

      {/* PDF Preview Overlay */}
      {selectedDoc && (
        <div className="preview-overlay" onClick={handleClosePreview}>
          <div className="preview-panel" onClick={(e) => e.stopPropagation()}>
            <PDFViewer
              docId={selectedDoc.doc_id}
              chunkId={selectedDoc.chunk_id}
              pageNum={selectedDoc.page_num}
              onClose={handleClosePreview}
              title={selectedDoc.title}
            />
          </div>
        </div>
      )}
    </div>
  );
}

export default App;

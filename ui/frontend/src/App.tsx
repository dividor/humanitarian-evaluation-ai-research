import React, { useState, useEffect } from 'react';
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

  // Load facets on mount
  useEffect(() => {
    loadFacets();
  }, []);

  const loadFacets = async () => {
    try {
      const response = await axios.get<Facets>(`${API_BASE_URL}/facets`);
      setFacets(response.data);
    } catch (error) {
      console.error('Error loading facets:', error);
    }
  };

  const handleSearch = async (e: React.FormEvent) => {
    e.preventDefault();
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
  };

  const handleFilterChange = (key: keyof SearchFilters, value: string) => {
    setFilters(prev => ({ ...prev, [key]: value }));
  };

  return (
    <div className="app">
      {/* Header */}
      <header className="header">
        <h1>🌍 Humanitarian Evaluation Search</h1>
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
      </header>

      <div className="main-content">
        {/* Filters Sidebar */}
        <aside className="filters">
          <h3>Filters</h3>

          {facets && (
            <>
              <div className="filter-group">
                <label>Organization</label>
                <select
                  value={filters.organization || ''}
                  onChange={(e) => handleFilterChange('organization', e.target.value)}
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
                <label>Year</label>
                <select
                  value={filters.year || ''}
                  onChange={(e) => handleFilterChange('year', e.target.value)}
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
                <label>Evaluation Type</label>
                <select
                  value={filters.evaluation_type || ''}
                  onChange={(e) => handleFilterChange('evaluation_type', e.target.value)}
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
                onClick={() => setFilters({})}
                className="clear-filters"
              >
                Clear Filters
              </button>
            </>
          )}
        </aside>

        {/* Results */}
        <main className="results">
          {results.length > 0 && (
            <div className="results-header">
              Found {results.length} results for "{query}"
            </div>
          )}

          {results.map((result) => (
            <div key={result.chunk_id} className="result-card">
              <h3 onClick={() => setSelectedDoc(result)} style={{ cursor: 'pointer' }}>
                {result.title}
              </h3>
              <div className="result-meta">
                {result.organization && <span className="badge">{result.organization}</span>}
                {result.year && <span className="badge">{result.year}</span>}
                <span className="badge">Page {result.page_num}</span>
                <span className="score">Score: {result.score.toFixed(3)}</span>
              </div>
              {result.headings.length > 0 && (
                <div className="breadcrumb">
                  {result.headings.join(' › ')}
                </div>
              )}
              <p className="snippet">{result.text.substring(0, 300)}...</p>
            </div>
          ))}

          {results.length === 0 && query && !loading && (
            <div className="no-results">
              No results found. Try different search terms or filters.
            </div>
          )}

          {!query && (
            <div className="welcome">
              <h2>Welcome to Humanitarian Evaluation Search</h2>
              <p>Search through indexed evaluation reports using semantic search.</p>
              <p>Try searching for topics like "climate change", "education", "health", etc.</p>
            </div>
          )}
        </main>

        {/* PDF Viewer */}
        {selectedDoc && (
          <aside className="pdf-viewer">
            <PDFViewer
              docId={selectedDoc.doc_id}
              pageNum={selectedDoc.page_num}
              searchText={selectedDoc.text}
              onClose={() => setSelectedDoc(null)}
              title={selectedDoc.title}
            />
          </aside>
        )}
      </div>
    </div>
  );
}

export default App;

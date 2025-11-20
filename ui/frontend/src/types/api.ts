export interface SearchResult {
  chunk_id: string;
  doc_id: string;
  text: string;
  page_num: number;
  headings: string[];
  score: number;
  title: string;
  organization?: string;
  year?: string;
  year_published?: string;
  evaluation_type?: string;
  country?: string;
  region?: string;
  theme?: string;
}

export interface SearchResponse {
  results: SearchResult[];
  total: number;
  query: string;
  filters?: Record<string, string[]>;
}

export interface FacetValue {
  value: string;
  count: number;
}

export interface Facets {
  organizations: FacetValue[];
  years: FacetValue[];
  evaluation_types: FacetValue[];
  countries: FacetValue[];
  regions: FacetValue[];
  themes: FacetValue[];
}

export interface HighlightBox {
  page: number;
  bbox: {
    l: number;
    r: number;
    t: number;
    b: number;
  };
  text: string;
}

export interface HighlightResponse {
  highlights: HighlightBox[];
  total: number;
}

export interface SearchFilters {
  organization?: string;
  year?: string;
  evaluation_type?: string;
  country?: string;
  region?: string;
  theme?: string;
}

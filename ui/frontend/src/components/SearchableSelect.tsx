import React, { useState, useRef, useEffect } from 'react';
import '../App.css';

interface SearchableSelectProps {
  label: string;
  options: { value: string; count: number }[];
  selectedValues: string[];
  onChange: (values: string[]) => void;
  placeholder?: string;
}

export const SearchableSelect: React.FC<SearchableSelectProps> = ({
  label,
  options,
  selectedValues,
  onChange,
  placeholder = 'Search...'
}) => {
  const [isOpen, setIsOpen] = useState(false);
  const [searchTerm, setSearchTerm] = useState('');
  const containerRef = useRef<HTMLDivElement>(null);

  // Filter options based on search term
  const filteredOptions = options.filter(option =>
    option.value.toLowerCase().includes(searchTerm.toLowerCase())
  );

  // Close dropdown when clicking outside
  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (containerRef.current && !containerRef.current.contains(event.target as Node)) {
        setIsOpen(false);
      }
    };

    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, []);

  const toggleOption = (value: string) => {
    if (selectedValues.includes(value)) {
      onChange(selectedValues.filter(v => v !== value));
    } else {
      onChange([...selectedValues, value]);
    }
  };

  const removeSelected = (value: string) => {
    onChange(selectedValues.filter(v => v !== value));
  };

  const clearAll = () => {
    onChange([]);
  };

  return (
    <div className="filter-group searchable-select-container" ref={containerRef}>
      <label className="filter-label">{label}</label>

      {/* Selected items */}
      {selectedValues.length > 0 && (
        <div className="selected-items">
          {selectedValues.map(value => {
            const option = options.find(o => o.value === value);
            return (
              <div key={value} className="selected-item">
                <span className="selected-item-text">
                  {value.substring(0, 30)}{value.length > 30 ? '...' : ''}
                  {option && ` (${option.count})`}
                </span>
                <button
                  className="selected-item-remove"
                  onClick={() => removeSelected(value)}
                  aria-label={`Remove ${value}`}
                >
                  ×
                </button>
              </div>
            );
          })}
          <button className="clear-all-button" onClick={clearAll}>
            Clear all
          </button>
        </div>
      )}

      {/* Search input */}
      <div className="searchable-select-input-wrapper">
        <input
          type="text"
          className="searchable-select-input"
          placeholder={placeholder}
          value={searchTerm}
          onChange={(e) => setSearchTerm(e.target.value)}
          onFocus={() => setIsOpen(true)}
        />
        <button
          className="searchable-select-toggle"
          onClick={() => setIsOpen(!isOpen)}
          aria-label="Toggle dropdown"
        >
          {isOpen ? '▲' : '▼'}
        </button>
      </div>

      {/* Dropdown */}
      {isOpen && (
        <div className="searchable-select-dropdown">
          {filteredOptions.length === 0 ? (
            <div className="searchable-select-no-results">No results found</div>
          ) : (
            filteredOptions.map(option => (
              <div
                key={option.value}
                className={`searchable-select-option ${selectedValues.includes(option.value) ? 'selected' : ''}`}
                onClick={() => toggleOption(option.value)}
              >
                <input
                  type="checkbox"
                  checked={selectedValues.includes(option.value)}
                  onChange={() => {}}
                  className="searchable-select-checkbox"
                />
                <span className="searchable-select-option-text">
                  {option.value}
                </span>
                <span className="searchable-select-option-count">({option.count})</span>
              </div>
            ))
          )}
        </div>
      )}
    </div>
  );
};

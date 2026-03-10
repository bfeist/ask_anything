import { useEffect, useLayoutEffect, useRef } from "react";

interface Props {
  results: SearchResult[];
  onSelect: (result: SearchResult) => void;
  selectedId: number | null;
  isSearching: boolean;
  query: string;
  onSelectedElementChange?: (el: HTMLButtonElement | null) => void;
  onScrollContainerChange?: (el: HTMLDivElement | null) => void;
}

function formatTime(seconds: number | null): string {
  if (seconds === null || seconds === undefined) return "—";
  const m = Math.floor(seconds / 60);
  const s = Math.floor(seconds % 60);
  return `${m}:${s.toString().padStart(2, "0")}`;
}

function scoreBar(score: number): string {
  // Visual score indicator using block characters
  const filled = Math.round(score * 20);
  return "█".repeat(filled) + "░".repeat(20 - filled);
}

/**
 * Displays search results as a scrollable list.
 */
export default function ResultsList({
  results,
  onSelect,
  selectedId,
  isSearching,
  query,
  onSelectedElementChange,
  onScrollContainerChange,
}: Props): React.JSX.Element {
  const listRef = useRef<HTMLDivElement>(null);
  const itemRefs = useRef<Map<number, HTMLButtonElement>>(new Map());

  useEffect(() => {
    onScrollContainerChange?.(listRef.current);
    return () => onScrollContainerChange?.(null);
  }, [onScrollContainerChange]);

  // No automatic scroll on selection — the parent handles window scrolling.

  // Notify parent so it can draw the connector line.
  useLayoutEffect(() => {
    if (!onSelectedElementChange) return;
    if (selectedId === null) {
      onSelectedElementChange(null);
      return;
    }
    onSelectedElementChange(itemRefs.current.get(selectedId) ?? null);
  }, [selectedId, results, onSelectedElementChange]);

  if (!query) {
    return <div className="results-empty">Enter a search query to find matching questions.</div>;
  }

  if (isSearching) {
    return <div className="results-empty">Searching…</div>;
  }

  if (results.length === 0 && query.length > 0) {
    return <div className="results-empty">No matching questions found.</div>;
  }

  return (
    <div className="results-list" ref={listRef}>
      {results.map((r, i) => (
        <button
          key={r.question.id}
          ref={(el) => {
            if (el) itemRefs.current.set(r.question.id, el);
            else itemRefs.current.delete(r.question.id);
          }}
          className={`result-item ${selectedId === r.question.id ? "result-item-selected" : ""}`}
          onClick={() => onSelect(r)}
          type="button"
        >
          <div className="result-rank">{i + 1}</div>
          <div className="result-body">
            <div className="result-text">{r.question.text}</div>
            <div className="result-meta">
              <span className="result-score" title={`Similarity: ${r.score.toFixed(3)}`}>
                {scoreBar(r.score)} {(r.score * 100).toFixed(1)}%
              </span>
              <span className="result-event">{r.question.event_type.replace(/_/g, " ")}</span>
              <span className="result-time">{formatTime(r.question.question_start)}</span>
            </div>
          </div>
        </button>
      ))}
    </div>
  );
}

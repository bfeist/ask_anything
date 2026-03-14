import { useEffect, useLayoutEffect, useRef } from "react";
import styles from "./ResultsList.module.css";

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

function ScoreBar({ score }: { score: number }): React.JSX.Element {
  const pct = Math.round(score * 100);
  return (
    <div className={styles.scoreBarWrap} title={`Similarity: ${score.toFixed(3)}`}>
      <div className={styles.scoreBar}>
        <div className={styles.scoreBarFill} style={{ width: `${pct}%` }} />
      </div>
      <span className={styles.scoreBarLabel}>{(score * 100).toFixed(1)}%</span>
    </div>
  );
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
    return (
      <div className={styles.resultsEmpty}>Enter a search query to find matching questions.</div>
    );
  }

  if (isSearching) {
    return <div className={styles.resultsEmpty}>Searching…</div>;
  }

  if (results.length === 0 && query.length > 0) {
    return <div className={styles.resultsEmpty}>No matching questions found.</div>;
  }

  return (
    <div className={styles.resultsList} ref={listRef}>
      {results.map((r, i) => (
        <button
          key={r.question.id}
          ref={(el) => {
            if (el) itemRefs.current.set(r.question.id, el);
            else itemRefs.current.delete(r.question.id);
          }}
          className={`${styles.resultItem} ${selectedId === r.question.id ? styles.resultItemSelected : ""}`}
          onClick={() => onSelect(r)}
          type="button"
        >
          <div className={styles.resultRank}>{i + 1}</div>
          <div className={styles.resultBody}>
            <div className={styles.resultText}>{r.question.text}</div>
            <div className={styles.resultMeta}>
              <ScoreBar score={r.score} />
              <span className={styles.resultEvent}>{r.question.event_type.replace(/_/g, " ")}</span>
              <span>{formatTime(r.question.question_start)}</span>
            </div>
          </div>
        </button>
      ))}
    </div>
  );
}

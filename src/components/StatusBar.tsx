interface Props {
  messages: InitProgress[];
  indexReady: boolean;
  modelReady: boolean;
  questionCount?: number;
}

/**
 * Shows loading progress while the search engine initialises.
 */
export default function StatusBar({
  messages,
  indexReady,
  modelReady,
  questionCount,
}: Props): React.JSX.Element {
  // Once everything is ready, show a minimal status
  if (indexReady && modelReady) {
    const countText = questionCount ? ` with ${questionCount.toLocaleString()} questions` : "";
    return (
      <div className="status-bar status-bar--ready">Ready — Semantic search active{countText}</div>
    );
  }

  const latest = messages[messages.length - 1];
  return (
    <div className="status-bar status-bar--loading">
      <div className="status-spinner" />
      <span>{latest?.message ?? "Initialising…"}</span>
      <div className="status-checks">
        <span className={indexReady ? "status-check--done" : ""}>
          {indexReady ? "✓" : "○"} Index
        </span>
        <span className={modelReady ? "status-check--done" : ""}>
          {modelReady ? "✓" : "○"} Model
        </span>
      </div>
    </div>
  );
}

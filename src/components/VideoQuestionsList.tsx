import { useEffect, useMemo, useRef } from "react";
import { getQuestionsForVideo } from "@/lib/searchEngine";
import styles from "./VideoQuestionsList.module.css";

interface Props {
  /** The source_file identifier for the current video. */
  sourceFile: string;
  /** The currently active question ID (highlighted in the list). */
  activeQuestionId: number;
  /** Called when the user clicks a different question. */
  onSeek: (question: IndexQuestion) => void;
}

function formatTime(seconds: number | null): string {
  if (seconds === null || seconds === undefined) return "—";
  const m = Math.floor(seconds / 60);
  const s = Math.floor(seconds % 60);
  return `${m}:${s.toString().padStart(2, "0")}`;
}

/**
 * Displays all Q&A pairs from a single video, highlighting the
 * currently-playing question. Clicking a row seeks the video.
 */
export default function VideoQuestionsList({
  sourceFile,
  activeQuestionId,
  onSeek,
}: Props): React.JSX.Element {
  const questions = useMemo(() => getQuestionsForVideo(sourceFile), [sourceFile]);
  const activeRef = useRef<HTMLButtonElement>(null);
  const listRef = useRef<HTMLDivElement>(null);

  // Scroll the active question into view within the list container only
  useEffect(() => {
    const container = listRef.current;
    const activeEl = activeRef.current;
    if (!container || !activeEl) return;

    const cRect = container.getBoundingClientRect();
    const iRect = activeEl.getBoundingClientRect();

    const relTop = iRect.top - cRect.top;
    const relBottom = iRect.bottom - cRect.top;

    if (relTop < 0) {
      container.scrollTo({ top: container.scrollTop + relTop, behavior: "smooth" });
    } else if (relBottom > container.clientHeight) {
      container.scrollTo({
        top: container.scrollTop + (relBottom - container.clientHeight),
        behavior: "smooth",
      });
    }
  }, [activeQuestionId]);

  if (questions.length <= 1) {
    return <></>;
  }

  return (
    <div className={styles.videoQuestions}>
      <div className={styles.videoQuestionsHeader}>
        Questions in this video ({questions.length})
      </div>
      <div className={styles.videoQuestionsList} ref={listRef}>
        {questions.map((q, i) => {
          const isActive = q.id === activeQuestionId;
          return (
            <button
              key={q.id}
              ref={isActive ? activeRef : null}
              className={`${styles.vqItem} ${isActive ? styles.vqItemActive : ""}`}
              onClick={() => onSeek(q)}
              type="button"
            >
              <div className={styles.vqRank}>{i + 1}</div>
              <div className={styles.vqBody}>
                <div className={styles.vqText}>{q.text}</div>
                <div className={styles.vqMeta}>
                  <span>{formatTime(q.question_start)}</span>
                  <span className={styles.vqEvent}>{q.event_type.replace(/_/g, " ")}</span>
                </div>
              </div>
            </button>
          );
        })}
      </div>
    </div>
  );
}

import { useMemo } from "react";
import { fnv1a32, hashToUnitFloat, yearFraction } from "@/utils/timelineUtils";
import styles from "./QuestionsTimeline.module.css";

interface Props {
  questions: IndexQuestion[];
  /** All questions in the index — shown as small grey background dots. */
  allQuestions?: IndexQuestion[];
  activeQuestionId?: number | null;
  /** Used to keep the pseudo-random layout stable per video. */
  seedKey: string;
  onSelectQuestionId?: (questionId: number) => void;
  startYear?: number;
  endYear?: number;
  /** Real date metadata keyed by .qa.json filename. When provided, dots use actual dates. */
  videoDates?: VideoDates | null;
}

export default function QuestionsTimeline({
  questions,
  allQuestions = [],
  activeQuestionId = null,
  seedKey,
  onSelectQuestionId,
  startYear = 2000,
  endYear = 2025,
  videoDates = null,
}: Props): React.JSX.Element {
  const { dots, bgDots, years } = useMemo(() => {
    // Derive the range from videoDates when available.
    let resolvedStartYear = startYear;
    let resolvedEndYear = endYear;
    if (videoDates) {
      let minMs = Infinity;
      let maxMs = -Infinity;
      for (const entry of Object.values(videoDates)) {
        if (entry.date) {
          const ms = Date.parse(entry.date);
          if (!isNaN(ms)) {
            if (ms < minMs) minMs = ms;
            if (ms > maxMs) maxMs = ms;
          }
        }
      }
      if (isFinite(minMs) && isFinite(maxMs)) {
        resolvedStartYear = new Date(minMs).getUTCFullYear();
        resolvedEndYear = new Date(maxMs).getUTCFullYear();
      }
    }

    const startMs = Date.UTC(resolvedStartYear, 0, 1, 0, 0, 0);
    // End of resolvedEndYear (inclusive)
    const endMs = Date.UTC(resolvedEndYear, 11, 31, 23, 59, 59);

    const resultIds = new Set(questions.map((q) => q.id));
    const resultSourceFiles = new Set(questions.map((q) => q.source_file));

    // --- Anti-crowding for background dots ---
    // Step 1: one representative question per source_file (avoids stacking
    //         all questions from the same video at the exact same position).
    const seenSourceFiles = new Set<string>();
    const uniqueByVideo = allQuestions.filter((q) => {
      if (resultIds.has(q.id) || resultSourceFiles.has(q.source_file)) return false;
      if (seenSourceFiles.has(q.source_file)) return false;
      seenSourceFiles.add(q.source_file);
      return true;
    });

    // Step 2: compute position and spatially thin — keep at most one dot per
    //         bin of width `BG_MIN_SPACING_PCT` to prevent crowding.
    const BG_MIN_SPACING_PCT = 1.2; // ~12px on a 1000px bar
    const positioned = uniqueByVideo.map((q) => {
      let frac: number;
      const qaKey = q.source_file.replace(/\.qa_text\.json$/, ".qa.json");
      const entry = videoDates?.[qaKey];
      if (entry?.date) {
        const dateMs = Date.parse(entry.date);
        frac = yearFraction(dateMs, startMs, endMs);
      } else {
        const h = fnv1a32(`${seedKey}|${q.id}|${q.text}`);
        const r = hashToUnitFloat(h);
        const dateMs = startMs + Math.floor(r * (endMs - startMs));
        frac = yearFraction(dateMs, startMs, endMs);
      }
      return { id: q.id, xPct: frac * 100, label: q.text };
    });

    positioned.sort((a, b) => a.xPct - b.xPct);
    const bgDots: typeof positioned = [];
    let lastKeptXPct = -Infinity;
    for (const dot of positioned) {
      if (dot.xPct - lastKeptXPct >= BG_MIN_SPACING_PCT) {
        bgDots.push(dot);
        lastKeptXPct = dot.xPct;
      }
    }

    const dots = questions.map((q) => {
      let frac: number;

      const qaKey = q.source_file.replace(/\.qa_text\.json$/, ".qa.json");
      const entry = videoDates?.[qaKey];
      if (entry?.date) {
        const dateMs = Date.parse(entry.date);
        frac = yearFraction(dateMs, startMs, endMs);
      } else {
        // Fall back to deterministic pseudo-random placement.
        const h = fnv1a32(`${seedKey}|${q.id}|${q.text}`);
        const r = hashToUnitFloat(h);
        const dateMs = startMs + Math.floor(r * (endMs - startMs));
        frac = yearFraction(dateMs, startMs, endMs);
      }

      return { id: q.id, xPct: frac * 100, label: q.text };
    });

    const years = [] as Array<{ year: number; xPct: number; isMajor: boolean }>;
    for (let y = resolvedStartYear; y <= resolvedEndYear; y++) {
      const frac = (y - resolvedStartYear) / Math.max(1, resolvedEndYear - resolvedStartYear);
      years.push({
        year: y,
        xPct: frac * 100,
        isMajor: y === resolvedStartYear || y === resolvedEndYear || y % 5 === 0,
      });
    }

    return { dots, bgDots, years };
  }, [questions, allQuestions, seedKey, startYear, endYear, videoDates]);

  return (
    <div className={styles.questionsTimeline}>
      <div className={styles.qtTrack}>
        <div className={styles.qtLine} />

        {years.map((y) => (
          <div
            key={y.year}
            className={`${styles.qtYearTick} ${y.isMajor ? styles.qtYearTickMajor : ""}`}
            style={{ left: `${y.xPct}%` }}
          >
            <div className={styles.qtYearLabel}>{y.year}</div>
          </div>
        ))}

        {bgDots.map((d) => (
          <button
            key={d.id}
            type="button"
            className={styles.qtDot}
            style={{ left: `${d.xPct}%` }}
            onClick={() => onSelectQuestionId?.(d.id)}
            aria-label={`View video: ${d.label}`}
          />
        ))}

        {dots.map((d) => (
          <button
            key={d.id}
            type="button"
            className={`${styles.qtDot} ${styles.qtDotResult}${d.id === activeQuestionId ? ` ${styles.qtDotActive}` : ""}`}
            style={{ left: `${d.xPct}%` }}
            onClick={() => onSelectQuestionId?.(d.id)}
            aria-label={`View video: ${d.label}`}
          />
        ))}
      </div>
    </div>
  );
}

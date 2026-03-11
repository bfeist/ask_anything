import { useMemo } from "react";

interface Props {
  questions: IndexQuestion[];
  activeQuestionId?: number | null;
  /** Used to keep the pseudo-random layout stable per video. */
  seedKey: string;
  onSelectQuestionId?: (questionId: number) => void;
  startYear?: number;
  endYear?: number;
  /** Real date metadata keyed by .qa.json filename. When provided, dots use actual dates. */
  videoDates?: VideoDates | null;
}

function clamp01(v: number): number {
  return Math.min(1, Math.max(0, v));
}

/* eslint-disable no-bitwise */
// Small, deterministic hash (FNV-1a 32-bit).
function fnv1a32(input: string): number {
  let hash = 0x811c9dc5;
  for (let i = 0; i < input.length; i++) {
    hash ^= input.charCodeAt(i);
    // hash *= 16777619 (but keep 32-bit)
    hash = (hash + (hash << 1) + (hash << 4) + (hash << 7) + (hash << 8) + (hash << 24)) >>> 0;
  }
  return hash >>> 0;
}

function hashToUnitFloat(hash: number): number {
  // Map uint32 -> [0,1)
  return (hash >>> 0) / 2 ** 32;
}
/* eslint-enable no-bitwise */

function yearFraction(dateMs: number, startMs: number, endMs: number): number {
  if (endMs <= startMs) return 0;
  return clamp01((dateMs - startMs) / (endMs - startMs));
}

export default function QuestionsTimeline({
  questions,
  activeQuestionId = null,
  seedKey,
  onSelectQuestionId,
  startYear = 2000,
  endYear = 2025,
  videoDates = null,
}: Props): React.JSX.Element {
  const { dots, years } = useMemo(() => {
    const startMs = Date.UTC(startYear, 0, 1, 0, 0, 0);
    // End of endYear (inclusive)
    const endMs = Date.UTC(endYear, 11, 31, 23, 59, 59);

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
    for (let y = startYear; y <= endYear; y++) {
      const frac = (y - startYear) / Math.max(1, endYear - startYear);
      years.push({
        year: y,
        xPct: frac * 100,
        isMajor: y === startYear || y === endYear || y % 5 === 0,
      });
    }

    return { dots, years };
  }, [questions, seedKey, startYear, endYear, videoDates]);

  return (
    <div className="questions-timeline">
      <div className="qt-track">
        <div className="qt-line" />

        {years.map((y) => (
          <div
            key={y.year}
            className={`qt-year-tick ${y.isMajor ? "qt-year-tick--major" : ""}`}
            style={{ left: `${y.xPct}%` }}
          >
            <div className="qt-year-label">{y.year}</div>
          </div>
        ))}

        {dots.map((d) => (
          <button
            key={d.id}
            type="button"
            className={`qt-dot ${d.id === activeQuestionId ? "qt-dot--active" : ""}`}
            style={{ left: `${d.xPct}%` }}
            onClick={() => onSelectQuestionId?.(d.id)}
            aria-label={`View video: ${d.label}`}
          />
        ))}
      </div>
    </div>
  );
}

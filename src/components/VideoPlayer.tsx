import { useCallback, useRef, useState } from "react";
interface Props {
  result: SearchResult | null;
  onClose: () => void;
}

function formatTime(seconds: number): string {
  const m = Math.floor(seconds / 60);
  const s = Math.floor(seconds % 60);
  return `${m}:${s.toString().padStart(2, "0")}`;
}

/**
 * Inner player component — receives a `key` so React unmounts/remounts
 * it when the selected result changes, which naturally resets all state.
 */
function VideoPlayerInner({
  result,
  onClose,
}: {
  result: SearchResult;
  onClose: () => void;
}): React.JSX.Element {
  const videoRef = useRef<HTMLVideoElement>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(0);

  const seekTarget = result.question.question_start ?? 0;

  const handleLoaded = useCallback(() => {
    setIsLoading(false);
    setError(null);
    const video = videoRef.current;
    if (video && seekTarget > 0) {
      video.currentTime = seekTarget;
    }
  }, [seekTarget]);

  const handleError = useCallback(() => {
    setIsLoading(false);
    const video = videoRef.current;
    const code = video?.error?.code;
    const msg = video?.error?.message;
    setError(`Video error${code ? ` (code ${code})` : ""}: ${msg || "Failed to load"}`);
  }, []);

  const handleTimeUpdate = useCallback(() => {
    if (videoRef.current) {
      setCurrentTime(videoRef.current.currentTime);
    }
  }, []);

  const handleDurationChange = useCallback(() => {
    if (videoRef.current) {
      setDuration(videoRef.current.duration);
    }
  }, []);

  const q = result.question;
  const hasAnswers = q.answers.length > 0;
  const firstAnswer = q.answers[0];

  return (
    <div className="video-player">
      <div className="video-header">
        <div className="video-question-text">{q.text}</div>
        <button className="video-close" onClick={onClose} type="button">
          ×
        </button>
      </div>

      {error ? (
        <div className="video-error">
          <p>{error}</p>
          <p className="video-error-hint">Video file: {result.videoUrl.split("/").pop()}</p>
        </div>
      ) : (
        <div className="video-container">
          {isLoading && <div className="video-loading">Loading video…</div>}
          <video
            ref={videoRef}
            className="video-element"
            src={result.videoUrl}
            controls
            onLoadedData={handleLoaded}
            onError={handleError}
            onTimeUpdate={handleTimeUpdate}
            onDurationChange={handleDurationChange}
            preload="auto"
            autoPlay={true}
          >
            <track kind="captions" />
          </video>
        </div>
      )}

      <div className="video-info">
        <div className="video-info-row">
          <span className="video-info-label">Current:</span>
          <span>{formatTime(currentTime)}</span>
          <span className="video-info-sep">/</span>
          <span>{formatTime(duration)}</span>
        </div>
        <div className="video-info-row">
          <span className="video-info-label">Question:</span>
          <span>
            {formatTime(q.question_start ?? 0)} – {formatTime(q.question_end ?? 0)}
          </span>
          {q.question_start !== null && (
            <button
              className="video-seek-btn"
              onClick={() => {
                if (videoRef.current && q.question_start !== null) {
                  videoRef.current.currentTime = q.question_start;
                }
              }}
              type="button"
            >
              ↻ Seek to Q
            </button>
          )}
        </div>
        {hasAnswers && (
          <div className="video-info-row">
            <span className="video-info-label">Answer:</span>
            <span>
              {formatTime(firstAnswer.start)} – {formatTime(firstAnswer.end)}
            </span>
            <button
              className="video-seek-btn"
              onClick={() => {
                if (videoRef.current) {
                  videoRef.current.currentTime = firstAnswer.start;
                }
              }}
              type="button"
            >
              ↻ Seek to A
            </button>
          </div>
        )}
        <div className="video-info-row">
          <span className="video-info-label">Event type:</span>
          <span>{q.event_type.replace(/_/g, " ")}</span>
        </div>
        <div className="video-info-row">
          <span className="video-info-label">Score:</span>
          <span>{(result.score * 100).toFixed(1)}% match</span>
        </div>
      </div>
    </div>
  );
}

/**
 * Outer wrapper: uses `key` to force remount when the question changes,
 * which resets all internal state cleanly without useEffect + setState.
 */
export default function VideoPlayer({ result, onClose }: Props): React.JSX.Element | null {
  if (!result) return null;
  return <VideoPlayerInner key={result.question.id} result={result} onClose={onClose} />;
}

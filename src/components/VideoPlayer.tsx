import { useCallback, useMemo, useRef, useState } from "react";
import { FontAwesomeIcon } from "@fortawesome/react-fontawesome";
import { faXmark, faPlay, faPause } from "@fortawesome/free-solid-svg-icons";
import VideoQuestionsList from "@/components/VideoQuestionsList";
import { getQuestionsForVideo } from "@/lib/searchEngine";
import styles from "./VideoPlayer.module.css";

interface Props {
  result: SearchResult | null;
  onClose: () => void;
  panelRef?: React.Ref<HTMLDivElement>;
}

function formatTime(seconds: number): string {
  const m = Math.floor(seconds / 60);
  const s = Math.floor(seconds % 60);
  return `${m}:${s.toString().padStart(2, "0")}`;
}

/**
 * Inner player component — receives a `key` so React unmounts/remounts
 * it when the video changes, which naturally resets all state.
 * Supports intra-video question navigation without remounting.
 */
function VideoPlayerInner({
  result,
  onClose,
}: {
  result: SearchResult;
  onClose: () => void;
}): React.JSX.Element {
  const videoRef = useRef<HTMLVideoElement>(null);
  const hideTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(0);
  const [isPaused, setIsPaused] = useState(false);
  const [controlsVisible, setControlsVisible] = useState(false);
  const [activeQuestionId, setActiveQuestionId] = useState(result.question.id);

  // All questions for this video, sorted by question_start.
  const videoQuestions = useMemo(
    () => getQuestionsForVideo(result.question.source_file),
    [result.question.source_file]
  );

  const activeQuestion = videoQuestions.find((q) => q.id === activeQuestionId) ?? result.question;

  const seekTarget = result.question.question_start ?? 0;

  const handleLoaded = useCallback(() => {
    setIsLoading(false);
    setError(null);
    const video = videoRef.current;
    if (!video) return;
    if (seekTarget > 0) {
      video.currentTime = seekTarget;
    }
    video.play().catch(() => {});
  }, [seekTarget]);

  const handlePlayPause = useCallback(() => {
    const video = videoRef.current;
    if (!video) return;
    if (video.paused) {
      video.play().catch(() => {});
    } else {
      video.pause();
    }
  }, []);

  const handleScrub = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    if (videoRef.current) {
      videoRef.current.currentTime = Number(e.target.value);
    }
  }, []);

  const handleError = useCallback(() => {
    setIsLoading(false);
    const video = videoRef.current;
    const code = video?.error?.code;
    const msg = video?.error?.message;
    setError(`Video error${code ? ` (code ${code})` : ""}: ${msg || "Failed to load"}`);
  }, []);

  const handleTimeUpdate = useCallback(() => {
    const video = videoRef.current;
    if (!video) return;
    const t = video.currentTime;
    setCurrentTime(t);

    // Find the question whose window contains the current playback time.
    // A question "owns" the time from its question_start up to the next
    // question's question_start (or end of video).
    let best: IndexQuestion | null = null;
    for (const q of videoQuestions) {
      if (q.question_start !== null && q.question_start <= t) {
        best = q;
      }
    }
    if (best !== null && best.id !== activeQuestionId) {
      setActiveQuestionId(best.id);
    }
  }, [videoQuestions, activeQuestionId]);

  const handleDurationChange = useCallback(() => {
    if (videoRef.current) {
      setDuration(videoRef.current.duration);
    }
  }, []);

  const handlePlay = useCallback(() => setIsPaused(false), []);
  const handlePause = useCallback(() => setIsPaused(true), []);

  const handleContainerPointerDown = useCallback((e: React.PointerEvent<HTMLDivElement>) => {
    if (e.pointerType === "touch") {
      setControlsVisible((visible) => {
        if (visible) {
          if (hideTimerRef.current !== null) clearTimeout(hideTimerRef.current);
          return false;
        }
        if (hideTimerRef.current !== null) clearTimeout(hideTimerRef.current);
        hideTimerRef.current = setTimeout(() => setControlsVisible(false), 4000);
        return true;
      });
    }
  }, []);

  const handleSeekToQuestion = useCallback((question: IndexQuestion) => {
    const video = videoRef.current;
    if (video && question.question_start !== null) {
      video.currentTime = question.question_start;
      video.play().catch(() => {});
    }
  }, []);

  const q = activeQuestion;
  const hasAnswers = q.answers.length > 0;
  const firstAnswer = q.answers[0];

  return (
    <div className={styles.videoPlayer}>
      <div className={styles.videoHeader}>
        <div className={styles.videoQuestionText}>{q.text}</div>
        <button className={styles.videoClose} onClick={onClose} type="button" aria-label="Close">
          <FontAwesomeIcon icon={faXmark} />
        </button>
      </div>

      {error ? (
        <div className={styles.videoError}>
          <p>{error}</p>
          <p className={styles.videoErrorHint}>Video file: {result.videoUrl.split("/").pop()}</p>
        </div>
      ) : (
        <div
          className={`${styles.videoContainer}${controlsVisible ? ` ${styles.controlsVisible}` : ""}`}
          onPointerDown={handleContainerPointerDown}
        >
          {isLoading && <div className={styles.videoLoading}>Loading video…</div>}
          <video
            ref={videoRef}
            className={styles.videoElement}
            src={result.videoUrl}
            playsInline
            onLoadedData={handleLoaded}
            onError={handleError}
            onTimeUpdate={handleTimeUpdate}
            onDurationChange={handleDurationChange}
            onPlay={handlePlay}
            onPause={handlePause}
            preload="auto"
          >
            <track kind="captions" />
          </video>
          <div className={styles.videoControls} onPointerDown={(e) => e.stopPropagation()}>
            <button
              className={styles.videoPlayBtn}
              onClick={handlePlayPause}
              type="button"
              aria-label={isPaused ? "Play" : "Pause"}
            >
              <FontAwesomeIcon icon={isPaused ? faPlay : faPause} />
            </button>
            <input
              className={styles.videoScrubber}
              type="range"
              min={0}
              max={duration || 100}
              value={currentTime}
              step={0.5}
              onChange={handleScrub}
            />
            <span className={styles.videoTimeDisplay}>
              {formatTime(currentTime)} / {formatTime(duration)}
            </span>
          </div>
        </div>
      )}

      <div className={styles.videoInfo}>
        <div className={styles.videoInfoRow}>
          <span className={styles.videoInfoLabel}>Question:</span>
          <span>
            {formatTime(q.question_start ?? 0)} – {formatTime(q.question_end ?? 0)}
          </span>
          {q.question_start !== null && (
            <button
              className={styles.videoSeekBtn}
              onClick={() => {
                if (videoRef.current && q.question_start !== null) {
                  videoRef.current.currentTime = q.question_start;
                }
              }}
              type="button"
            >
              ↻ Seek to Question
            </button>
          )}
        </div>
        {hasAnswers && (
          <div className={styles.videoInfoRow}>
            <span className={styles.videoInfoLabel}>Answer:</span>
            <span>
              {formatTime(firstAnswer.start)} – {formatTime(firstAnswer.end)}
            </span>
            <button
              className={styles.videoSeekBtn}
              onClick={() => {
                if (videoRef.current) {
                  videoRef.current.currentTime = firstAnswer.start;
                }
              }}
              type="button"
            >
              ↻ Seek to Answer
            </button>
          </div>
        )}
        <div className={styles.videoInfoRow}>
          <span className={styles.videoInfoLabel}>Event type:</span>
          <span>{q.event_type.replace(/_/g, " ")}</span>
        </div>
      </div>

      {q.answer_text && (
        <div className={styles.videoAnswer}>
          <div className={styles.videoAnswerLabel}>Answer transcript</div>
          <p className={styles.videoAnswerText}>{q.answer_text}</p>
        </div>
      )}

      <VideoQuestionsList
        sourceFile={result.question.source_file}
        activeQuestionId={activeQuestionId}
        onSeek={handleSeekToQuestion}
      />
    </div>
  );
}

/**
 * Outer wrapper: uses `key` on the video URL to force remount only when the
 * video file changes. Intra-video question navigation is handled internally.
 */
export default function VideoPlayer({
  result,
  onClose,
  panelRef,
}: Props): React.JSX.Element | null {
  if (!result) return null;
  return (
    <div className={styles.videoPanel} ref={panelRef}>
      <VideoPlayerInner key={result.videoUrl} result={result} onClose={onClose} />
    </div>
  );
}

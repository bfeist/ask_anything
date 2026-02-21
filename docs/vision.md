# Project Vision: Ask Anything

## Purpose

Build a searchable database of questions and answers from NASA astronaut Q&A sessions,
interviews, and educational events. The primary source material is video footage from
the Internet Archive where students from schools across the United States talk to
astronauts aboard the International Space Station.

## End-User Experience

A user will be able to:

1. **Ask any question** in natural language (e.g. "How do you sleep in space?").
2. The system finds the **closest matching question** that a student actually asked
   an astronaut in a real Q&A session.
3. The interface **jumps to the exact moment** a kid asks that question in the video,
   then plays the astronaut's answer — and only that answer.
4. **Search for any phrase** across all transcribed video content to find where
   astronauts discuss specific topics, even outside formal Q&A segments.

## Source Material Characteristics

The videos have a distinctive structure:

- **Formal Q&A events**: A student introduces themselves ("Hi, my name is X"),
  asks their pre-defined question, then there is a ~5–7 second satellite delay
  before the astronaut begins answering. One video can contain 10–20+ questions.
- **Interviews**: Astronauts on the ground or in space being interviewed by media
  or educators. Topics vary; these are less structured.
- **Ambient / operational footage**: General ISS operations, chatter, or briefings
  that may contain interesting subject matter but not formal Q&A.

## Data Model (Target)

Each video ultimately produces:

| Field          | Description                                       |
| -------------- | ------------------------------------------------- |
| `video_file`   | Source video filename                             |
| `video_path`   | Full path to the video                            |
| `segment_type` | `question`, `answer`, `intro`, `closing`, `other` |
| `speaker`      | Speaker label (from diarization) or inferred role |
| `text`         | Transcribed text of the segment                   |
| `start`        | Start time in seconds (aligned)                   |
| `end`          | End time in seconds (aligned)                     |
| `question_id`  | Links a question to its answer (for Q&A pairs)    |

## Pipeline Stages

| Stage | Script                     | Status | Description                                                    |
| ----- | -------------------------- | ------ | -------------------------------------------------------------- |
| 1     | `1_scan_ia_metadata.py`    | Done   | Scan Internet Archive for NASA video metadata                  |
| 2     | `2_classify_candidates.py` | Done   | Use LLM to classify which videos are relevant                  |
| 3     | `3_download_lowres.py`     | Done   | Download low-res versions of relevant videos                   |
| 4     | `4_transcribe_videos.py`   | Done   | Transcribe with WhisperX + alignment + diarization             |
| 5     | _(planned)_                | —      | Extract Q&A pairs: detect questions, match answers, assign IDs |
| 6     | _(planned)_                | —      | Build searchable index (embeddings + full-text)                |
| 7     | _(planned)_                | —      | Web interface for question search and video playback           |

## Stage 4: Transcription Details

**Technology**: WhisperX with `large-v3` model on CUDA GPU.

**Process**:

1. Load audio from video via ffmpeg.
2. Transcribe with Whisper (batched inference on GPU).
3. Force-align timestamps for English using wav2vec2 alignment model.
4. _(Optional)_ Speaker diarization via pyannote.audio to label who is speaking.

**Output per video**: `data/transcripts/<video_stem>.json`

```json
{
  "video_file": "..._lowres.mp4",
  "model": "large-v3",
  "language": "en",
  "diarization": true,
  "segments": [
    {
      "start": 162.03,
      "end": 164.60,
      "text": "Hi, my name is Wajida.",
      "speaker": "SPEAKER_02",
      "words": [
        {"word": "Hi,", "start": 162.03, "end": 162.29, "score": 0.95},
        ...
      ]
    }
  ]
}
```

## Stage 5: Q&A Extraction (Planned)

The next stage will analyze transcript segments to:

1. **Detect question boundaries**: Look for patterns like "Hi, my name is X" followed
   by "my question is..." to identify where questions start.
2. **Detect answer boundaries**: Use the ~6s satellite delay gap + diarization speaker
   change to find where answers begin. Answers end when the astronaut finishes speaking
   (next gap or next question intro).
3. **Pair questions and answers**: Link each question to its corresponding answer.
4. **Classify non-Q&A content**: Label introductions, closing remarks, and other
   discussion segments separately but still retain them for full-text search.

## Key Design Decisions

- **Word-level timestamps**: Kept in output so the UI can highlight words during playback.
- **Full segments retained**: Even non-Q&A content (intros, closings, operational chatter)
  is preserved so users can search for any phrase across all videos.
- **Diarization is valuable but optional**: The script works without it; diarization
  helps distinguish kid voices from astronaut voices for Q&A pairing.
- **Incremental processing**: Each stage skips already-processed items, so the pipeline
  can be re-run safely after adding new videos.

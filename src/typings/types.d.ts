/** A single answer timing block within a Q&A pair. */
interface AnswerTiming {
  start: number;
  end: number;
}

/** A single question from the search index. */
interface IndexQuestion {
  id: number;
  text: string;
  source_file: string;
  event_type: string;
  pair_index: number;
  question_start: number | null;
  question_end: number | null;
  answers: AnswerTiming[];
}

/** Metadata for the search index. */
interface IndexMeta {
  version: number;
  model: string;
  embedding_dim: number;
  embedding_dtype: string;
  num_questions: number;
  min_question_words: number;
  built_at: string;
  files: {
    questions: string;
    embeddings: string;
  };
}

/** A search result returned from the engine. */
interface SearchResult {
  question: IndexQuestion;
  score: number;
  videoUrl: string;
}

/** Progress event emitted during search-engine initialisation. */
interface InitProgress {
  stage: "index" | "model";
  message: string;
  done?: boolean;
}

/**
 * Thin callable type for the transformers.js feature-extraction pipeline.
 * Avoids the overly-complex union that pipeline()'s overloaded return type
 * produces, which TypeScript cannot represent.
 */
type Extractor = (
  text: string,
  options?: Record<string, unknown>
) => Promise<{ data: ArrayLike<number> }>;

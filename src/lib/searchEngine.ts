/**
 * Semantic search engine that runs entirely in the browser.
 *
 * Loads the pre-built search index (questions.json + embeddings.bin) and
 * runs a sentence-transformer model (all-MiniLM-L6-v2) via ONNX /
 * transformers.js to embed queries and compute cosine similarity.
 */

// ---------------------------------------------------------------------------
// Paths (relative to dev server root — Vite middleware serves these)
// ---------------------------------------------------------------------------
const INDEX_META_URL = "/data/search_index/index_meta.json";
const QUESTIONS_URL = "/data/search_index/questions.json";
const EMBEDDINGS_URL = "/data/search_index/embeddings.bin";

// ---------------------------------------------------------------------------
// Singleton state
// ---------------------------------------------------------------------------
let _meta: IndexMeta | null = null;
let _questions: IndexQuestion[] | null = null;
let _embeddings: Float32Array | null = null; // (num_questions × dim) row-major
let _extractor: Extractor | null = null;

let _initPromise: Promise<void> | null = null;
let _modelPromise: Promise<void> | null = null;

// ---------------------------------------------------------------------------
// Initialisation — index data (fast) and model (slower, separate)
// ---------------------------------------------------------------------------

/**
 * Load the search index (metadata + questions + embeddings).
 * This is fast — just a few MB of JSON + binary.
 */
export async function loadIndex(onProgress?: (p: InitProgress) => void): Promise<void> {
  if (_meta && _questions && _embeddings) return;
  if (_initPromise) return _initPromise;

  _initPromise = (async () => {
    onProgress?.({ stage: "index", message: "Loading search index…" });

    // Fetch metadata and questions in parallel
    const [metaRes, questionsRes] = await Promise.all([
      fetch(INDEX_META_URL),
      fetch(QUESTIONS_URL),
    ]);

    _meta = (await metaRes.json()) as IndexMeta;
    _questions = (await questionsRes.json()) as IndexQuestion[];

    onProgress?.({
      stage: "index",
      message: `Loaded ${_meta.num_questions} questions. Fetching embeddings…`,
    });

    // Fetch binary embeddings
    const embRes = await fetch(EMBEDDINGS_URL);
    const embBuf = await embRes.arrayBuffer();

    // The Python pipeline stores float16—widen to float32 for dot-product math
    const dim = _meta.embedding_dim;
    const numQuestions = _meta.num_questions;

    if (_meta.embedding_dtype === "float16") {
      const f16 = new Uint16Array(embBuf);
      const f32 = new Float32Array(f16.length);
      for (let i = 0; i < f16.length; i++) {
        f32[i] = float16ToFloat32(f16[i]);
      }
      _embeddings = f32;
    } else {
      _embeddings = new Float32Array(embBuf);
    }

    // Sanity check
    if (_embeddings.length !== numQuestions * dim) {
      throw new Error(
        `Embeddings size mismatch: got ${_embeddings.length}, expected ${numQuestions * dim}`
      );
    }

    onProgress?.({
      stage: "index",
      message: `Search index ready (${numQuestions} questions, ${dim}d)`,
      done: true,
    });
  })();

  return _initPromise;
}

/**
 * Load the sentence-transformer model for query embedding.
 * This can take 5-15 s on first load (model download + ONNX init).
 */
export async function loadModel(onProgress?: (p: InitProgress) => void): Promise<void> {
  if (_extractor) return;
  if (_modelPromise) return _modelPromise;

  _modelPromise = (async () => {
    onProgress?.({
      stage: "model",
      message: "Loading ML model (all-MiniLM-L6-v2)…",
    });

    const { pipeline } = await import("@huggingface/transformers");
    _extractor = (await pipeline("feature-extraction", "Xenova/all-MiniLM-L6-v2", {
      dtype: "fp32",
    })) as unknown as Extractor;

    onProgress?.({
      stage: "model",
      message: "Model ready",
      done: true,
    });
  })();

  return _modelPromise;
}

/** Convenience: load both index and model. */
export async function init(onProgress?: (p: InitProgress) => void): Promise<void> {
  await Promise.all([loadIndex(onProgress), loadModel(onProgress)]);
}

// ---------------------------------------------------------------------------
// Search
// ---------------------------------------------------------------------------

/**
 * Derive the Vite dev-server video URL from a `source_file` field.
 *
 * `source_file` looks like:
 *   `<ia_id>__<video_stem>_lowres.qa_text.json`
 *
 * The actual mp4 on disk is:
 *   `D:/ask_anything_ia_videos_raw/<ia_id>__<video_stem>_lowres.mp4`
 *
 * Vite middleware serves that directory at `/videos/`.
 */
function videoUrlFromSource(sourceFile: string): string {
  const videoFilename = sourceFile.replace(/\.qa_text\.json$/, ".mp4");
  return `/videos/${encodeURIComponent(videoFilename)}`;
}

/**
 * Embed a query string and return the normalised float32 vector.
 */
async function embedQuery(query: string): Promise<Float32Array> {
  if (!_extractor) throw new Error("Model not loaded—call loadModel() first");

  const output = await _extractor(query, {
    pooling: "mean",
    normalize: true,
  });

  // output.data is a Float32Array of shape [1, dim] or flat [dim]
  return new Float32Array(output.data as Float32Array);
}

/**
 * Search the index with a natural-language query.
 *
 * Returns the top `k` results above `minScore`.
 */
export async function search(
  query: string,
  { k = 20, minScore = 0.15 }: { k?: number; minScore?: number } = {}
): Promise<SearchResult[]> {
  if (!_embeddings || !_questions || !_meta) {
    throw new Error("Index not loaded—call loadIndex() first");
  }
  if (!_extractor) {
    throw new Error("Model not loaded—call loadModel() first");
  }

  const queryVec = await embedQuery(query);
  const dim = _meta.embedding_dim;
  const numQ = _meta.num_questions;

  // Compute cosine similarity (dot product since both are unit-normalised)
  const scores = new Float32Array(numQ);
  for (let i = 0; i < numQ; i++) {
    let dot = 0;
    const offset = i * dim;
    for (let j = 0; j < dim; j++) {
      dot += queryVec[j] * _embeddings[offset + j];
    }
    scores[i] = dot;
  }

  // Build a sorted top-k list
  const indices = Array.from({ length: numQ }, (_, i) => i);
  indices.sort((a, b) => scores[b] - scores[a]);

  const results: SearchResult[] = [];
  for (const idx of indices) {
    if (results.length >= k) break;
    if (scores[idx] < minScore) break;

    const question = _questions[idx];
    results.push({
      question,
      score: scores[idx],
      videoUrl: videoUrlFromSource(question.source_file),
    });
  }

  return results;
}

// ---------------------------------------------------------------------------
// Status helpers
// ---------------------------------------------------------------------------

export function isIndexLoaded(): boolean {
  return _meta !== null && _questions !== null && _embeddings !== null;
}

export function isModelLoaded(): boolean {
  return _extractor !== null;
}

export function isReady(): boolean {
  return isIndexLoaded() && isModelLoaded();
}

export function getQuestionCount(): number {
  return _meta?.num_questions ?? 0;
}

// ---------------------------------------------------------------------------
// IEEE 754 float16 → float32 conversion
// ---------------------------------------------------------------------------

/**
 * Convert an IEEE-754 float16 (stored in a uint16) to float32.
 *
 * Uses DataView to perform the conversion without bitwise operators
 * (the project's ESLint config has `no-bitwise` enabled).
 */
function float16ToFloat32(h: number): number {
  // Use a DataView for the bit reinterpretation
  const buf = new ArrayBuffer(4);
  const view = new DataView(buf);

  // Decode float16 fields via arithmetic (no bitwise ops)
  const sign = Math.floor(h / 32768); // h >> 15
  const exponent = Math.floor(h / 1024) % 32; // (h >> 10) & 0x1f
  const mantissa = h % 1024; // h & 0x3ff

  let f32Bits: number;

  if (exponent === 0) {
    if (mantissa === 0) {
      // ±Zero
      f32Bits = sign * 2147483648; // sign << 31
    } else {
      // Subnormal: convert to normalized float32
      let m = mantissa;
      let e = -14 + 127; // float32 bias adjusted
      // Shift mantissa up until the implicit leading 1 appears
      while (m < 1024) {
        m *= 2;
        e -= 1;
      }
      m -= 1024; // remove the implicit 1
      f32Bits = sign * 2147483648 + e * 8388608 + Math.floor(m * 8192); // mantissa scaled from 10 to 23 bits
    }
  } else if (exponent === 31) {
    // Infinity or NaN
    f32Bits = sign * 2147483648 + 255 * 8388608 + (mantissa !== 0 ? 1 : 0);
  } else {
    // Normal: rebias exponent from float16 (bias 15) to float32 (bias 127)
    const f32Exp = exponent - 15 + 127;
    f32Bits = sign * 2147483648 + f32Exp * 8388608 + Math.floor(mantissa * 8192);
  }

  view.setUint32(0, f32Bits, false);
  return view.getFloat32(0, false);
}

import { useCallback, useEffect, useRef, useState } from "react";
import SearchInput from "@/components/SearchInput";
import ResultsList from "@/components/ResultsList";
import VideoPlayer from "@/components/VideoPlayer";
import StatusBar from "@/components/StatusBar";
import { init, search, isIndexLoaded, isModelLoaded, getQuestionCount } from "@/lib/searchEngine";

function App(): React.JSX.Element {
  const [results, setResults] = useState<SearchResult[]>([]);
  const [selectedResult, setSelectedResult] = useState<SearchResult | null>(null);
  const [query, setQuery] = useState("");
  const [isSearching, setIsSearching] = useState(false);
  const [indexReady, setIndexReady] = useState(false);
  const [modelReady, setModelReady] = useState(false);
  const [statusMessages, setStatusMessages] = useState<InitProgress[]>([]);

  // Track the latest search request to avoid stale results
  const searchIdRef = useRef(0);

  // Initialise the search engine on mount
  useEffect(() => {
    const onProgress = (p: InitProgress) => {
      setStatusMessages((prev) => [...prev, p]);
      if (p.stage === "index" && p.done) setIndexReady(true);
      if (p.stage === "model" && p.done) setModelReady(true);
    };

    // Load index first (fast), then model (slower) — both in parallel
    init(onProgress).catch((err) => {
      console.error("Search engine init failed:", err);
      setStatusMessages((prev) => [...prev, { stage: "model", message: `Error: ${err.message}` }]);
    });
  }, []);

  const handleSearch = useCallback(async (newQuery: string) => {
    setQuery(newQuery);

    if (!newQuery.trim()) {
      setResults([]);
      return;
    }

    if (!isIndexLoaded() || !isModelLoaded()) {
      // Queue will be handled once ready
      return;
    }

    const id = ++searchIdRef.current;
    setIsSearching(true);

    try {
      const hits = await search(newQuery.trim(), { k: 25, minScore: 0.15 });
      // Only apply if this is still the latest search
      if (id === searchIdRef.current) {
        setResults(hits);
      }
    } catch (err) {
      console.error("Search error:", err);
      if (id === searchIdRef.current) {
        setResults([]);
      }
    } finally {
      if (id === searchIdRef.current) {
        setIsSearching(false);
      }
    }
  }, []);

  // Re-run the pending query once the engine is ready
  useEffect(() => {
    if (indexReady && modelReady && query.trim()) {
      handleSearch(query);
    }
  }, [indexReady, modelReady]); // eslint-disable-line react-hooks/exhaustive-deps

  const handleSelect = useCallback((result: SearchResult) => {
    setSelectedResult(result);
  }, []);

  const handleCloseVideo = useCallback(() => {
    setSelectedResult(null);
  }, []);

  const engineReady = indexReady && modelReady;

  return (
    <div className="app">
      <header className="app-header">
        <h1 className="app-title">Ask an Astronaut Anything</h1>
        <p className="app-subtitle">
          Search across astronaut Q&amp;A recordings onboard the International Space Station
        </p>
      </header>

      <StatusBar
        messages={statusMessages}
        indexReady={indexReady}
        modelReady={modelReady}
        questionCount={getQuestionCount()}
      />

      <main className="app-main">
        <div className="search-panel">
          <SearchInput
            onSearch={handleSearch}
            disabled={!engineReady}
            placeholder="Ask your question"
          />
          <ResultsList
            results={results}
            onSelect={handleSelect}
            selectedId={selectedResult?.question.id ?? null}
            isSearching={isSearching}
            query={query}
          />
        </div>

        <div className="player-panel">
          {selectedResult ? (
            <VideoPlayer result={selectedResult} onClose={handleCloseVideo} />
          ) : (
            <div className="player-placeholder">
              <p>Select a question to watch the response</p>
            </div>
          )}
        </div>
      </main>
    </div>
  );
}

export default App;

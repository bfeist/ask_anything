import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { gsap } from "gsap";
import headerImg1 from "@/images/Fb39yuyEuLtH2W5k9BT7LA.jpg";
import headerImg2 from "@/images/alexanderGerstWaving.webp";
import headerImg3 from "@/images/egr-xqtw4aez5m2-1.jpg";
import headerImg4 from "@/images/STARLINER-NASA-ASTRONAUTS-ISS.png";
import headerImg5 from "@/images/astronaut_with_fruit.jpg";
import headerImg6 from "@/images/iss061e006682.webp";
import headerImg7 from "@/images/wilmore-hague-williams.webp";
import astronautSvg from "@/images/astronaut1.svg";
import astronaut2Svg from "@/images/astronaut2.svg";
import SearchInput from "@/components/SearchInput";
import ResultsList from "@/components/ResultsList";
import QuestionsTimeline from "@/components/QuestionsTimeline";
import VideoPlayer from "@/components/VideoPlayer";
import ThemeToggle from "@/components/ThemeToggle";
import { init, search, isIndexLoaded, isModelLoaded } from "@/lib/searchEngine";

const HEADER_IMAGES = [headerImg1, headerImg2, headerImg3, headerImg4, headerImg5, headerImg6, headerImg7];

const QUESTION_POOL = [
  "What is it like to sleep in space?",
  "How do you exercise on the ISS?",
  "What do astronauts eat in space?",
  "How do you use the bathroom in space?",
  "What does Earth look like from space?",
  "How do you deal with being away from family?",
  "What is the hardest part of living in space?",
  "How long does it take to get to the ISS?",
  "What happens to your body in microgravity?",
  "Can you see stars from the ISS?",
  "What do you miss most about Earth?",
  "How do you stay mentally healthy in space?",
  "What is the view like from the cupola?",
  "How do you wash your hair in space?",
  "What experiments are you running on the station?",
  "How do spacewalks work?",
  "What is the biggest danger of living in space?",
  "How do you keep track of time in orbit?",
  "What does it feel like to return to Earth?",
  "How do you communicate with your family from space?",
];

function App(): React.JSX.Element {
  const [results, setResults] = useState<SearchResult[]>([]);
  const [selectedResult, setSelectedResult] = useState<SearchResult | null>(null);
  const [query, setQuery] = useState("");
  const [isSearching, setIsSearching] = useState(false);
  const [indexReady, setIndexReady] = useState(false);
  const [modelReady, setModelReady] = useState(false);
  const [statusMessages, setStatusMessages] = useState<InitProgress[]>([]);
  const [hasSearched, setHasSearched] = useState(false);
  const [suggestedQuery, setSuggestedQuery] = useState<string | undefined>(undefined);
  const [currentImageIndex, setCurrentImageIndex] = useState(() => Math.floor(Math.random() * HEADER_IMAGES.length));

  const commonQuestions = useMemo(() => {
    const pool = QUESTION_POOL.slice();
    for (let i = 0; i < 6; i++) {
      const j = i + Math.floor(Math.random() * (pool.length - i));
      [pool[i], pool[j]] = [pool[j], pool[i]];
    }
    return pool.slice(0, 6);
  }, []);

  useEffect(() => {
    const handleScroll = () => {
      document.documentElement.style.setProperty("--scroll-y", `${window.scrollY}px`);
    };
    window.addEventListener("scroll", handleScroll, { passive: true });
    return () => window.removeEventListener("scroll", handleScroll);
  }, []);

  useEffect(() => {
    if (hasSearched) return;
    const interval = setInterval(() => {
      setCurrentImageIndex((prev) => (prev + 1) % HEADER_IMAGES.length);
    }, 10000);
    return () => clearInterval(interval);
  }, [hasSearched]);

  // Ref so handleSearch (memoised with no deps) can read hasSearched without
  // needing it in its dependency array.
  const hasSearchedRef = useRef(false);

  const headerRef = useRef<HTMLElement>(null);
  const resultsPanelRef = useRef<HTMLDivElement>(null);
  const playerPanelRef = useRef<HTMLDivElement>(null);
  const timelineRowRef = useRef<HTMLDivElement>(null);

  const mainRef = useRef<HTMLElement>(null);
  const selectedQuestionElRef = useRef<HTMLButtonElement | null>(null);
  const resultsScrollElRef = useRef<HTMLDivElement | null>(null);
  const videoPanelElRef = useRef<HTMLDivElement | null>(null);

  const [connector, setConnector] = useState<{ w: number; h: number; d: string } | null>(null);

  // Track the latest search request to avoid stale results
  const searchIdRef = useRef(0);

  // Initialise the search engine on mount
  useEffect(() => {
    const handleScroll = () => {
      document.documentElement.style.setProperty("--scroll-y", `${window.scrollY}px`);
    };

    window.addEventListener("scroll", handleScroll, { passive: true });

    const onProgress = (p: InitProgress) => {
      setStatusMessages((prev) => [...prev, p]);
      if (p.stage === "index" && p.done) setIndexReady(true);
      if (p.stage === "model" && p.done) setModelReady(true);
    };

    init(onProgress).catch((err) => {
      console.error("Search engine init failed:", err);
      setStatusMessages((prev) => [...prev, { stage: "model", message: `Error: ${err.message}` }]);
    });

    return () => {
      window.removeEventListener("scroll", handleScroll);
    };
  }, []);

  // Animate hero → compact on first keystroke ----------------------------
  // We drive this imperatively so the hero class is still on the element
  // when we measure heroH — avoiding the React-render race.
  const animateHeroToCompact = useCallback(() => {
    const header = headerRef.current;
    if (!header) return;

    const heroH = header.offsetHeight;

    // Lock the hero title's font-size as an inline style BEFORE we touch any classes,
    // so the measurement step and React re-render can't collapse it.
    const heroTitleEl = header.querySelector(".app-title--hero") as HTMLElement | null;
    const compactTitleEl = header.querySelector(".app-title--compact") as HTMLElement | null;
    if (heroTitleEl) {
      const heroFontSize = parseFloat(getComputedStyle(heroTitleEl).fontSize);
      gsap.set(heroTitleEl, { fontSize: heroFontSize, letterSpacing: "-1.5px" });
    }

    // Temporarily remove the hero class to measure compact height
    header.classList.remove("app-header--hero");
    void header.offsetHeight; // force reflow
    const compactH = header.offsetHeight;

    // Restore + lock the hero height so GSAP can tween from it
    header.classList.add("app-header--hero");
    void header.offsetHeight;
    gsap.set(header, { height: heroH });

    gsap.to(header, {
      height: compactH,
      duration: 2,
      ease: "power3.inOut",
      onComplete: () => {
        header.classList.remove("app-header--hero");
        gsap.set(header, { clearProps: "height" });
      },
    });

    // Shrink the search input from hero size down to compact size.
    // Use fromTo so GSAP owns the values regardless of when React re-renders.
    // Values must match the CSS for .app-main--hero .search-input (from) and
    // .search-input (to) so clearProps doesn't cause a visual jump.
    const inputEl = mainRef.current?.querySelector(".search-input") as HTMLElement | null;
    if (inputEl) {
      gsap.fromTo(
        inputEl,
        { fontSize: 23, paddingTop: 18, paddingBottom: 18, paddingLeft: 22, paddingRight: 52 },
        {
          fontSize: 20,
          paddingTop: 16,
          paddingBottom: 16,
          paddingLeft: 28,
          paddingRight: 44,
          duration: 2,
          ease: "power3.inOut",
          onComplete: () => { gsap.set(inputEl, { clearProps: "all" }); },
        },
      );
    }

    // Hide the large hero title immediately, then fade the compact one in.
    if (heroTitleEl && compactTitleEl) {
      gsap.set(heroTitleEl, { opacity: 0 });
      // Pull hero title out of flow immediately (CSS backs this up for non-hero state).
      gsap.set(heroTitleEl, { position: "absolute", top: 0, left: 0 });
      // Compact title becomes in-flow element so subtitle sits right below it.
      gsap.set(compactTitleEl, { position: "relative", top: "auto", left: "auto" });
      gsap.fromTo(
        compactTitleEl,
        { opacity: 0 },
        {
          opacity: 1,
          duration: 1.5,
          delay: 0.75,
          ease: "power1.inOut",
          onComplete: () => { gsap.set(compactTitleEl, { clearProps: "all" }); },
        },
      );
    }

    // Fade subtitle in from invisible, same timing as the compact title.
    const subtitleEl = header.querySelector(".app-subtitle") as HTMLElement | null;
    if (subtitleEl) {
      gsap.set(subtitleEl, { opacity: 0 });
      gsap.fromTo(
        subtitleEl,
        { opacity: 0 },
        {
          opacity: 1,
          duration: 1.5,
          delay: 0.75,
          ease: "power1.inOut",
          onComplete: () => { gsap.set(subtitleEl, { clearProps: "all" }); },
        },
      );
    }
  }, []);

  // Fade + lift the content panels once they mount after hasSearched flips
  useEffect(() => {
    if (!hasSearched) return;

    const targets = [
      timelineRowRef.current,
      resultsPanelRef.current,
      playerPanelRef.current,
    ].filter(Boolean);

    gsap.fromTo(
      targets,
      { opacity: 0, y: 20 },
      { opacity: 1, y: 0, duration: 0.5, ease: "power2.out", delay: 0.55, stagger: 0.07 },
    );
  }, [hasSearched]);
  // -----------------------------------------------------------------------

  // Trigger the hero → compact animation on the first ever keystroke
  const handleFirstInput = useCallback(() => {
    if (!hasSearchedRef.current) {
      hasSearchedRef.current = true;
      // Run animation while hero class is still present, THEN flip state
      animateHeroToCompact();
      setHasSearched(true);
    }
  }, [animateHeroToCompact]);

  const handleSearch = useCallback(async (newQuery: string) => {
    setQuery(newQuery);

    if (!newQuery.trim()) {
      setResults([]);
      return;
    }

    if (!isIndexLoaded() || !isModelLoaded()) {
      return;
    }

    const id = ++searchIdRef.current;
    setIsSearching(true);

    try {
      const hits = await search(newQuery.trim(), { k: 25, minScore: 0.15 });
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

  // Scroll so the search input sits at the very top of the viewport.
  const scrollToSearchInput = useCallback(() => {
    // Use setTimeout to ensure the DOM has updated (e.g. video player rendered)
    // so the page is tall enough to scroll.
    setTimeout(() => {
      const el = mainRef.current?.querySelector(".search-input") as HTMLElement | null;
      if (!el) return;
      const top = el.getBoundingClientRect().top + window.scrollY - 8;
      window.scrollTo({ top, behavior: "smooth" });
    }, 50);
  }, []);

  const handleSelect = useCallback((result: SearchResult) => {
    setSelectedResult(result);
    scrollToSearchInput();
  }, [scrollToSearchInput]);

  const handleSelectFromTimeline = useCallback(
    (questionId: number) => {
      const match = results.find((r) => r.question.id === questionId) ?? null;
      if (match) {
        setSelectedResult(match);
        scrollToSearchInput();
      }
    },
    [results, scrollToSearchInput],
  );

  const handleCloseVideo = useCallback(() => {
    setSelectedResult(null);
  }, []);

  const engineReady = indexReady && modelReady;

  const recomputeConnector = useCallback(() => {
    const main = mainRef.current;
    const left = selectedQuestionElRef.current;
    const right = videoPanelElRef.current;

    if (!main || !left || !right) {
      setConnector(null);
      return;
    }

    const mainRect = main.getBoundingClientRect();
    const leftRect = left.getBoundingClientRect();
    const rightRect = right.getBoundingClientRect();

    const w = Math.max(1, Math.round(mainRect.width));
    const h = Math.max(1, Math.round(mainRect.height));

    const leftY = leftRect.top + Math.min(32, leftRect.height / 2);
    const rightY = rightRect.top + Math.min(32, rightRect.height / 2);

    const x1 = leftRect.right - mainRect.left + 1;
    const y1 = leftY - mainRect.top;
    const x2 = rightRect.left - mainRect.left - 1;
    const y2 = rightY - mainRect.top;

    const midX = (x1 + x2) / 2;

    const baseRadius = 18;
    const dx = Math.abs(x2 - x1);
    const dy = Math.abs(y2 - y1);
    const radius = Math.max(0, Math.min(baseRadius, dx / 4, dy / 2));
    const dir = y2 >= y1 ? 1 : -1;

    const d =
      radius > 0
        ? `M ${x1} ${y1} ` +
          `L ${midX - radius} ${y1} ` +
          `Q ${midX} ${y1} ${midX} ${y1 + dir * radius} ` +
          `L ${midX} ${y2 - dir * radius} ` +
          `Q ${midX} ${y2} ${midX + radius} ${y2} ` +
          `L ${x2} ${y2}`
        : `M ${x1} ${y1} L ${midX} ${y1} L ${midX} ${y2} L ${x2} ${y2}`;

    setConnector({ w, h, d });
  }, []);

  const handleSelectedElementChange = useCallback(
    (el: HTMLButtonElement | null) => {
      selectedQuestionElRef.current = el;
      recomputeConnector();
    },
    [recomputeConnector],
  );

  const handleResultsScrollContainerChange = useCallback(
    (el: HTMLDivElement | null) => {
      const prev = resultsScrollElRef.current;
      if (prev && prev !== el) {
        prev.removeEventListener("scroll", recomputeConnector);
      }
      resultsScrollElRef.current = el;
      if (el) {
        el.addEventListener("scroll", recomputeConnector, { passive: true });
      }
      recomputeConnector();
    },
    [recomputeConnector],
  );

  useEffect(() => {
    window.addEventListener("resize", recomputeConnector);
    window.addEventListener("scroll", recomputeConnector, { passive: true });
    return () => {
      window.removeEventListener("resize", recomputeConnector);
      window.removeEventListener("scroll", recomputeConnector);
    };
  }, [recomputeConnector]);

  useEffect(() => {
    recomputeConnector();
  }, [selectedResult, results, recomputeConnector]);

  useEffect(() => {
    if (typeof ResizeObserver === "undefined") return;
    const main = mainRef.current;
    if (!main) return;

    const ro = new ResizeObserver(() => recomputeConnector());
    ro.observe(main);
    if (videoPanelElRef.current) ro.observe(videoPanelElRef.current);
    if (selectedQuestionElRef.current) ro.observe(selectedQuestionElRef.current);

    return () => ro.disconnect();
  }, [recomputeConnector, selectedResult]);

  const hasSelection = useMemo(() => selectedResult !== null, [selectedResult]);

  const timelineQuestions = useMemo(() => results.map((r) => r.question), [results]);

  return (
    <div className="app-root">
      <header
        ref={headerRef}
        className={`app-header${hasSearched ? "" : " app-header--hero"}`}
      >
        {HEADER_IMAGES.map((img, index) => (
          <div
            key={img}
            className="app-header-bg"
            style={{
              backgroundImage: `url(${img})`,
              opacity: index === currentImageIndex ? 0.70 : 0,
            }}
          />
        ))}
        <div className="app-header-inner">
          <div className="app-header-text">
            <img src={astronaut2Svg} alt="" className="app-title-icon app-title-icon--hero" />
            <img src={astronaut2Svg} alt="" className="app-title-icon app-title-icon--compact" />
            <div className="app-header-text-content">
              <div className="app-title-wrap">
                <h1 className="app-title app-title--hero">Ask an Astronaut Anything</h1>
                <h1 className="app-title app-title--compact" aria-hidden="true">Ask an Astronaut Anything</h1>
              </div>
              <p className="app-subtitle">
                Search across astronaut Q&amp;A recordings onboard the International Space Station
              </p>
            </div>
          </div>
          <ThemeToggle />
        </div>
      </header>

      <div className="app">
        <main
          className={`app-main${hasSelection ? " app-main--has-selection" : ""}${hasSearched ? "" : " app-main--hero"}`}
          ref={mainRef}
        >
          <div className="search-row">
            <SearchInput
              onSearch={handleSearch}
              onFirstInput={handleFirstInput}
              disabled={!engineReady}
              placeholder="Ask your question..."
              statusText={!engineReady ? (statusMessages[statusMessages.length - 1]?.message ?? "Initialising…") : undefined}
              externalValue={suggestedQuery}
            />

            {!hasSearched && engineReady && (
              <div className="common-questions">
                <p className="common-questions-label">Common Questions:</p>
                <div className="common-questions-grid">
                  {commonQuestions.map((q) => (
                    <button
                      key={q}
                      className="common-question-pill"
                      onClick={() => setSuggestedQuery(q)}
                      type="button"
                    >
                      {q}
                    </button>
                  ))}
                </div>
              </div>
            )}

            {hasSearched && (
              <div ref={timelineRowRef}>
                <QuestionsTimeline
                  questions={timelineQuestions}
                  activeQuestionId={selectedResult?.question.id ?? null}
                  seedKey={query || "(empty)"}
                  onSelectQuestionId={handleSelectFromTimeline}
                  startYear={2000}
                  endYear={2025}
                />
              </div>
            )}
          </div>

          {hasSearched && (
            <div className="results-panel" ref={resultsPanelRef}>
              <ResultsList
                results={results}
                onSelect={handleSelect}
                selectedId={selectedResult?.question.id ?? null}
                isSearching={isSearching}
                query={query}
                onSelectedElementChange={handleSelectedElementChange}
                onScrollContainerChange={handleResultsScrollContainerChange}
              />
            </div>
          )}

          {hasSearched && (
            <div className="player-panel" ref={playerPanelRef}>
              {selectedResult ? (
                <VideoPlayer
                  result={selectedResult}
                  onClose={handleCloseVideo}
                  panelRef={videoPanelElRef}
                />
              ) : (
                <div className="player-placeholder">
                  <div className="placeholder-content">
                    <div className="astronaut-svg-container">
                      <img src={astronautSvg} alt="Astronaut illustration" className="astronaut-svg" />
                    </div>
                    <p>Select a question to watch the response</p>
                  </div>
                </div>
              )}
            </div>
          )}

          {connector && hasSelection && (
            <svg
              className="selection-connector"
              viewBox={`0 0 ${connector.w} ${connector.h}`}
              aria-hidden="true"
            >
              <path d={connector.d} />
            </svg>
          )}
        </main>
      </div>
    </div>
  );
}

export default App;

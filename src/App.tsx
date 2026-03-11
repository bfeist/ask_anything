import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { gsap } from "gsap";
import headerImg1 from "@/images/Fb39yuyEuLtH2W5k9BT7LA.jpg";
import headerImg2 from "@/images/alexanderGerstWaving.webp";
import headerImg3 from "@/images/egr-xqtw4aez5m2-1.jpg";
import headerImg4 from "@/images/STARLINER-NASA-ASTRONAUTS-ISS.png";
import headerImg5 from "@/images/astronaut_with_fruit.jpg";
import headerImg6 from "@/images/iss061e006682.webp";
import headerImg7 from "@/images/wilmore-hague-williams.webp";
import astronaut2Svg from "@/images/astronaut2.svg";
import CommonQuestions from "@/components/CommonQuestions";
import ConnectorSvg from "@/components/ConnectorSvg";
import PlayerPlaceholder from "@/components/PlayerPlaceholder";
import QuestionsTimeline from "@/components/QuestionsTimeline";
import ResultsList from "@/components/ResultsList";
import SearchInput from "@/components/SearchInput";
import ThemeToggle from "@/components/ThemeToggle";
import VideoPlayer from "@/components/VideoPlayer";
import { init, search, isIndexLoaded, isModelLoaded, getAllQuestions } from "@/lib/searchEngine";
import { useVideoDates } from "@/lib/useVideoDates";
import { computeConnectorPath, type ConnectorGeometry } from "@/utils/connector";
import { pickRandom } from "@/utils/pickRandom";
import styles from "./App.module.css";

const HEADER_IMAGES = [
  headerImg1,
  headerImg2,
  headerImg3,
  headerImg4,
  headerImg5,
  headerImg6,
  headerImg7,
];

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
  const [allIndexQuestions, setAllIndexQuestions] = useState<IndexQuestion[]>([]);
  const [suggestedQuery, setSuggestedQuery] = useState<string | undefined>(undefined);
  const videoDates = useVideoDates();
  const [currentImageIndex, setCurrentImageIndex] = useState(() =>
    Math.floor(Math.random() * HEADER_IMAGES.length)
  );

  const commonQuestions = useMemo(() => pickRandom(QUESTION_POOL, 6), []);

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

  const [connector, setConnector] = useState<ConnectorGeometry | null>(null);

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
      if (p.stage === "index" && p.done) {
        setIndexReady(true);
        setAllIndexQuestions(getAllQuestions());
      }
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
    const heroTitleEl = header.querySelector(`.${styles.appTitleHero}`) as HTMLElement | null;
    const compactTitleEl = header.querySelector(`.${styles.appTitleCompact}`) as HTMLElement | null;
    if (heroTitleEl) {
      const heroFontSize = parseFloat(getComputedStyle(heroTitleEl).fontSize);
      gsap.set(heroTitleEl, { fontSize: heroFontSize, letterSpacing: "-1.5px" });
    }

    // Temporarily remove the hero class to measure compact height
    header.classList.remove(styles.appHeaderHero);
    // eslint-disable-next-line @typescript-eslint/no-unused-expressions
    header.offsetHeight; // force reflow
    const compactH = header.offsetHeight;

    // Restore + lock the hero height so GSAP can tween from it
    header.classList.add(styles.appHeaderHero);
    // eslint-disable-next-line @typescript-eslint/no-unused-expressions
    header.offsetHeight; // force reflow
    gsap.set(header, { height: heroH });

    gsap.to(header, {
      height: compactH,
      duration: 2,
      ease: "power3.inOut",
      onComplete: () => {
        header.classList.remove(styles.appHeaderHero);
        gsap.set(header, { clearProps: "height" });
      },
    });

    // Shrink the search input from hero size down to compact size.
    // Use fromTo so GSAP owns the values regardless of when React re-renders.
    // Values must match the CSS for .searchInputHero (from) and
    // .searchInput (to) so clearProps doesn't cause a visual jump.
    const inputEl = mainRef.current?.querySelector("[data-search-input]") as HTMLElement | null;
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
          onComplete: () => {
            gsap.set(inputEl, { clearProps: "all" });
          },
        }
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
          onComplete: () => {
            gsap.set(compactTitleEl, { clearProps: "all" });
          },
        }
      );
    }

    // Fade subtitle in from invisible, same timing as the compact title.
    const subtitleEl = header.querySelector(`.${styles.appSubtitle}`) as HTMLElement | null;
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
          onComplete: () => {
            gsap.set(subtitleEl, { clearProps: "all" });
          },
        }
      );
    }
  }, []);

  // Animate compact → hero (reverse of animateHeroToCompact) ------------------
  // Called synchronously in handleClearInput, before React's batched render
  // adds app-header--hero back, so we can measure the compact height first.
  // GSAP inline styles override the CSS class values until clearProps is called.
  const animateCompactToHero = useCallback(() => {
    const header = headerRef.current;
    if (!header) return;

    const heroTitleEl = header.querySelector(`.${styles.appTitleHero}`) as HTMLElement | null;
    const compactTitleEl = header.querySelector(`.${styles.appTitleCompact}`) as HTMLElement | null;
    const subtitleEl = header.querySelector(`.${styles.appSubtitle}`) as HTMLElement | null;
    // Target the full row (icons + text) so astronaut icon is also covered.
    const headerTextEl = header.querySelector(`.${styles.appHeaderText}`) as HTMLElement | null;
    const inputEl = mainRef.current?.querySelector("[data-search-input]") as HTMLElement | null;

    [header, heroTitleEl, compactTitleEl, subtitleEl, headerTextEl, inputEl].forEach((el) => {
      if (el) gsap.killTweensOf(el);
    });

    // Hide synchronously BEFORE clearing individual inline styles.
    // React's setHasSearched(false) batches and re-renders after this handler
    // returns, so app-header--hero gets added back while headerTextEl is already
    // opacity:0 — preventing the instantaneous title-size snap from being visible.
    if (headerTextEl) gsap.set(headerTextEl, { opacity: 0 });

    // Now safe to clear per-element inline styles; everything is hidden above.
    if (heroTitleEl) gsap.set(heroTitleEl, { clearProps: "all" });
    if (compactTitleEl) gsap.set(compactTitleEl, { clearProps: "all" });
    if (subtitleEl) gsap.set(subtitleEl, { clearProps: "all" });

    const compactH = header.offsetHeight;

    // Temporarily add hero class to measure its height
    header.classList.add(styles.appHeaderHero);
    // eslint-disable-next-line @typescript-eslint/no-unused-expressions
    header.offsetHeight; // force reflow
    const heroH = header.offsetHeight;
    header.classList.remove(styles.appHeaderHero);
    // eslint-disable-next-line @typescript-eslint/no-unused-expressions
    header.offsetHeight; // force reflow

    // Lock at compact height so the GSAP tween controls it while React
    // re-renders (adding app-header--hero back).
    gsap.set(header, { height: compactH });

    const tl = gsap.timeline();

    // Expand header height + grow input simultaneously.
    tl.to(header, {
      height: heroH,
      duration: 2,
      ease: "power3.inOut",
      onComplete: () => {
        gsap.set(header, { clearProps: "height" });
      },
    });

    if (inputEl) {
      tl.fromTo(
        inputEl,
        { fontSize: 20, paddingTop: 16, paddingBottom: 16, paddingLeft: 28, paddingRight: 44 },
        {
          fontSize: 23,
          paddingTop: 18,
          paddingBottom: 18,
          paddingLeft: 22,
          paddingRight: 52,
          duration: 2,
          ease: "power3.inOut",
          onComplete: () => {
            gsap.set(inputEl, { clearProps: "all" });
          },
        },
        "<" // same time as header expansion
      );
    }

    // Fade the text+icon block back in near the end of the expansion.
    // By this point React has re-rendered with app-header--hero, so CSS
    // shows the hero title / hero icon automatically.
    if (headerTextEl) {
      tl.to(
        headerTextEl,
        {
          opacity: 1,
          duration: 1,
          ease: "power1.out",
          onComplete: () => {
            gsap.set(headerTextEl, { clearProps: "opacity" });
          },
        },
        "-=0.8"
      );
    }
  }, []);
  // ---------------------------------------------------------------------------

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
      { opacity: 1, y: 0, duration: 0.5, ease: "power2.out", delay: 0.55, stagger: 0.07 }
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

  const scrollToSearchInputOnMobile = useCallback(() => {
    if (window.matchMedia("(pointer: coarse)").matches) {
      // Only auto-scroll on mobile where the on-screen keyboard may cover the input
      setTimeout(() => {
        const el = mainRef.current?.querySelector("[data-search-input]") as HTMLElement | null;
        if (!el) return;
        const top = el.getBoundingClientRect().top + window.scrollY - 8;
        window.scrollTo({ top, behavior: "smooth" });
      }, 50);
    }
  }, []);

  const handleSelect = useCallback(
    (result: SearchResult) => {
      setSelectedResult(result);
      scrollToSearchInputOnMobile();
    },
    [scrollToSearchInputOnMobile]
  );

  const handleSelectFromTimeline = useCallback(
    (questionId: number) => {
      const match = results.find((r) => r.question.id === questionId) ?? null;
      if (match) {
        setSelectedResult(match);
        scrollToSearchInputOnMobile();
      }
    },
    [results, scrollToSearchInputOnMobile]
  );

  const handleCloseVideo = useCallback(() => {
    setSelectedResult(null);
  }, []);

  const handleClearInput = useCallback(() => {
    // Run the animation synchronously first — the DOM is still in compact state
    // because React batches all the setState calls below into a single render
    // that fires after this handler completes.
    animateCompactToHero();
    hasSearchedRef.current = false;
    setHasSearched(false);
    setResults([]);
    setSelectedResult(null);
    setQuery("");
    setSuggestedQuery(undefined);
  }, [animateCompactToHero]);

  const engineReady = indexReady && modelReady;

  const recomputeConnector = useCallback(() => {
    const main = mainRef.current;
    const left = selectedQuestionElRef.current;
    const right = videoPanelElRef.current;

    if (!main || !left || !right) {
      setConnector(null);
      return;
    }

    setConnector(
      computeConnectorPath(
        main.getBoundingClientRect(),
        left.getBoundingClientRect(),
        right.getBoundingClientRect()
      )
    );
  }, []);

  const handleSelectedElementChange = useCallback(
    (el: HTMLButtonElement | null) => {
      selectedQuestionElRef.current = el;
      recomputeConnector();
    },
    [recomputeConnector]
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
    [recomputeConnector]
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
    <div className={styles.appRoot}>
      <header
        ref={headerRef}
        className={`${styles.appHeader}${hasSearched ? "" : ` ${styles.appHeaderHero}`}`}
      >
        {HEADER_IMAGES.map((img, index) => (
          <div
            key={img}
            className={styles.appHeaderBg}
            style={{
              backgroundImage: `url(${img})`,
              opacity: index === currentImageIndex ? 0.7 : 0,
            }}
          />
        ))}
        <div className={styles.appHeaderInner}>
          <div className={styles.appHeaderText}>
            <img
              src={astronaut2Svg}
              alt=""
              className={`${styles.appTitleIcon} ${styles.appTitleIconHero}`}
            />
            <img
              src={astronaut2Svg}
              alt=""
              className={`${styles.appTitleIcon} ${styles.appTitleIconCompact}`}
            />
            <div className={styles.appHeaderTextContent}>
              <div className={styles.appTitleWrap}>
                <h1 className={`${styles.appTitle} ${styles.appTitleHero}`}>
                  Ask an Astronaut Anything
                </h1>
                <h1 className={`${styles.appTitle} ${styles.appTitleCompact}`} aria-hidden="true">
                  Ask an Astronaut Anything
                </h1>
              </div>
              <p className={styles.appSubtitle}>
                Search across astronaut Q&amp;A recordings onboard the International Space Station
              </p>
            </div>
          </div>
          <ThemeToggle />
        </div>
      </header>

      <div className={styles.app}>
        <main
          className={`${styles.appMain}${hasSearched ? "" : ` ${styles.appMainHero}`}`}
          ref={mainRef}
        >
          <div className={styles.searchRow}>
            <SearchInput
              onSearch={handleSearch}
              onFirstInput={handleFirstInput}
              onClear={handleClearInput}
              disabled={!engineReady}
              placeholder="Ask your question..."
              hero={!hasSearched}
              statusText={
                !engineReady
                  ? (statusMessages[statusMessages.length - 1]?.message ?? "Initialising…")
                  : undefined
              }
              externalValue={suggestedQuery}
            />

            {!hasSearched && engineReady && (
              <CommonQuestions questions={commonQuestions} onSelect={setSuggestedQuery} />
            )}

            {hasSearched && (
              <div ref={timelineRowRef}>
                <QuestionsTimeline
                  questions={timelineQuestions}
                  allQuestions={allIndexQuestions}
                  activeQuestionId={selectedResult?.question.id ?? null}
                  seedKey={query || "(empty)"}
                  onSelectQuestionId={handleSelectFromTimeline}
                  startYear={2000}
                  endYear={2025}
                  videoDates={videoDates}
                />
              </div>
            )}
          </div>

          {hasSearched && (
            <div className={styles.resultsPanel} ref={resultsPanelRef}>
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
            <div className={styles.playerPanel} ref={playerPanelRef}>
              {selectedResult ? (
                <VideoPlayer
                  result={selectedResult}
                  onClose={handleCloseVideo}
                  panelRef={videoPanelElRef}
                />
              ) : (
                <PlayerPlaceholder />
              )}
            </div>
          )}

          {connector && hasSelection && <ConnectorSvg connector={connector} />}
        </main>
      </div>
    </div>
  );
}

export default App;

import { useCallback, useEffect, useState } from "react";
import { FontAwesomeIcon } from "@fortawesome/react-fontawesome";
import { faMoon, faSun } from "@fortawesome/free-solid-svg-icons";
import styles from "./ThemeToggle.module.css";

type Theme = "dark" | "light";

const STORAGE_KEY = "ask-anything-theme";

function getStoredTheme(): Theme {
  try {
    const stored = localStorage.getItem(STORAGE_KEY);
    if (stored === "light" || stored === "dark") return stored;
  } catch {
    /* storage unavailable */
  }
  return "dark";
}

function applyTheme(theme: Theme): void {
  document.documentElement.setAttribute("data-theme", theme);
}

export default function ThemeToggle(): React.JSX.Element {
  const [theme, setTheme] = useState<Theme>(getStoredTheme);

  // Apply on mount and whenever theme changes
  useEffect(() => {
    applyTheme(theme);
    try {
      localStorage.setItem(STORAGE_KEY, theme);
    } catch {
      /* noop */
    }
  }, [theme]);

  const toggle = useCallback(() => {
    setTheme((prev) => (prev === "dark" ? "light" : "dark"));
  }, []);

  const isLight = theme === "light";

  return (
    <button
      className={styles.themeToggle}
      onClick={toggle}
      type="button"
      role="switch"
      aria-checked={isLight}
      aria-label={`Switch to ${isLight ? "dark" : "light"} mode`}
      title={`Switch to ${isLight ? "dark" : "light"} mode`}
    >
      <span
        className={`${styles.themeToggleIcon} ${styles.themeToggleIconMoon}`}
        aria-hidden="true"
      >
        <FontAwesomeIcon icon={faMoon} width={14} height={14} />
      </span>
      <span className={styles.themeToggleTrack}>
        <span className={styles.themeToggleThumb} />
      </span>
      <span className={`${styles.themeToggleIcon} ${styles.themeToggleIconSun}`} aria-hidden="true">
        <FontAwesomeIcon icon={faSun} width={14} height={14} />
      </span>
    </button>
  );
}

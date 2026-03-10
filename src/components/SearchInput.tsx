import { useCallback, useEffect, useRef, useState } from "react";

interface Props {
  onSearch: (query: string) => void;
  onFirstInput?: () => void;
  disabled?: boolean;
  placeholder?: string;
  statusText?: string;
  externalValue?: string;
}

/**
 * Debounced search input that fires `onSearch` as the user types.
 */
export default function SearchInput({
  onSearch,
  onFirstInput,
  disabled = false,
  placeholder = "Ask anything…",
  statusText,
  externalValue,
}: Props): React.JSX.Element {
  const [value, setValue] = useState("");
  const timerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const firstInputFiredRef = useRef(false);

  // Sync when a suggestion is chosen externally
  useEffect(() => {
    if (externalValue === undefined) return;
    setValue(externalValue);
    if (externalValue && !firstInputFiredRef.current) {
      firstInputFiredRef.current = true;
      onFirstInput?.();
    }
    onSearch(externalValue);
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [externalValue]);

  const debouncedSearch = useCallback(
    (text: string) => {
      if (timerRef.current) clearTimeout(timerRef.current);
      timerRef.current = setTimeout(() => {
        onSearch(text);
      }, 200);
    },
    [onSearch]
  );

  useEffect(() => {
    return () => {
      if (timerRef.current) clearTimeout(timerRef.current);
    };
  }, []);

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const text = e.target.value;
    setValue(text);
    // Fire onFirstInput immediately on the very first non-empty keystroke
    if (text && !firstInputFiredRef.current) {
      firstInputFiredRef.current = true;
      onFirstInput?.();
    }
    debouncedSearch(text);
  };

  const handleClear = () => {
    setValue("");
    onSearch("");
  };

  const handleBlur = () => {
    // On iOS Safari, when input loses focus, scroll to ensure
    // video player is visible (since iOS auto-scrolls inputs into view)
    if (window.innerWidth < 900) {
      setTimeout(() => {
        window.scrollTo(0, 0);
      }, 100);
    }
  };

  return (
    <div className="search-input-wrapper">
      <input
        type="text"
        className="search-input"
        value={value}
        onChange={handleChange}
        disabled={disabled}
        placeholder={placeholder}
        // eslint-disable-next-line jsx-a11y/no-autofocus
        autoFocus
        spellCheck={false}
        onBlur={handleBlur}
      />
      {statusText && !value && (
        <span className="search-input-status" aria-hidden="true">{statusText}</span>
      )}
      {value && (
        <button
          className="search-clear"
          onClick={handleClear}
          aria-label="Clear search"
          type="button"
        >
          ×
        </button>
      )}
    </div>
  );
}

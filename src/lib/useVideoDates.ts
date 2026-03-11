import { useEffect, useState } from "react";

const PROD_DATA = import.meta.env.VITE_PROD_DATA === "true";
const STATIC_ORIGIN = PROD_DATA ? "https://askanything.benfeist.com" : "";

const VIDEO_DATES_URL = `${STATIC_ORIGIN}/static_assets/data/video_dates.json`;

export function useVideoDates(): VideoDates | null {
  const [data, setData] = useState<VideoDates | null>(null);

  useEffect(() => {
    fetch(VIDEO_DATES_URL)
      .then((r) => r.json() as Promise<VideoDates>)
      .then(setData)
      .catch((err) => console.error("Failed to load video_dates.json:", err));
  }, []);

  return data;
}

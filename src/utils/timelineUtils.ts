export function clamp01(v: number): number {
  return Math.min(1, Math.max(0, v));
}

/* eslint-disable no-bitwise */
/** FNV-1a 32-bit hash — small and deterministic, used for stable pseudo-random dot placement. */
export function fnv1a32(input: string): number {
  let hash = 0x811c9dc5;
  for (let i = 0; i < input.length; i++) {
    hash ^= input.charCodeAt(i);
    // hash *= 16777619 (kept as 32-bit)
    hash = (hash + (hash << 1) + (hash << 4) + (hash << 7) + (hash << 8) + (hash << 24)) >>> 0;
  }
  return hash >>> 0;
}

/** Maps a uint32 hash value to [0, 1). */
export function hashToUnitFloat(hash: number): number {
  return (hash >>> 0) / 2 ** 32;
}
/* eslint-enable no-bitwise */

/** Returns the fractional position of `dateMs` within [startMs, endMs], clamped to [0, 1]. */
export function yearFraction(dateMs: number, startMs: number, endMs: number): number {
  if (endMs <= startMs) return 0;
  return clamp01((dateMs - startMs) / (endMs - startMs));
}

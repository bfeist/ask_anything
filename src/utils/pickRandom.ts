/**
 * Returns `count` randomly-sampled items from `pool` without replacement,
 * using a partial Fisher-Yates shuffle.  The original array is not mutated.
 */
export function pickRandom<T>(pool: T[], count: number): T[] {
  const copy = pool.slice();
  const n = Math.min(count, copy.length);
  for (let i = 0; i < n; i++) {
    const j = i + Math.floor(Math.random() * (copy.length - i));
    [copy[i], copy[j]] = [copy[j], copy[i]];
  }
  return copy.slice(0, n);
}

export interface ConnectorGeometry {
  w: number;
  h: number;
  d: string;
}

/**
 * Computes the SVG path geometry for the selection connector line drawn
 * between the selected question button and the video panel.
 *
 * Takes pre-measured DOMRects so the function is pure and testable without
 * a live DOM.
 */
export function computeConnectorPath(
  mainRect: DOMRect,
  leftRect: DOMRect,
  rightRect: DOMRect
): ConnectorGeometry {
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

  return { w, h, d };
}

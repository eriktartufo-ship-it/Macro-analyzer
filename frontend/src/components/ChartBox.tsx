import { ReactNode, useEffect, useRef, useState } from "react";

type Props = {
  aspect?: number;
  minHeight?: number;
  children: (size: { width: number; height: number }) => ReactNode;
};

/**
 * ChartBox responsive container — pattern Sport app 2026-05-28.
 *
 * Misura il proprio width via ResizeObserver e passa width+height numerici
 * al chart. Aspect ratio configurabile.
 *
 * Permette SVG charts custom di essere responsive senza ResponsiveContainer
 * libraries (Macro Analyzer = no library by design, CLAUDE.md).
 */
export function ChartBox({ aspect = 2.4, minHeight = 200, children }: Props) {
  const ref = useRef<HTMLDivElement>(null);
  const [width, setWidth] = useState(0);

  useEffect(() => {
    const el = ref.current;
    if (!el) return;

    const update = () => {
      const w = el.clientWidth;
      if (w > 0) setWidth(w);
    };

    update();
    const ro = new ResizeObserver(update);
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  const height = Math.max(minHeight, Math.round(width / aspect));

  return (
    <div ref={ref} style={{ width: "100%", minWidth: 0 }}>
      {width > 0 ? (
        children({ width, height })
      ) : (
        <div style={{ height: minHeight }} />
      )}
    </div>
  );
}

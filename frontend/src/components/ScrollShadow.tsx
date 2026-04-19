import { useEffect, useRef, useState, type CSSProperties, type ReactNode } from "react";

interface Props {
  children: ReactNode;
  className?: string;
  innerClassName?: string;
  innerStyle?: CSSProperties;
  axis?: "x" | "y";
}

export function ScrollShadow({
  children,
  className,
  innerClassName,
  innerStyle,
  axis = "x",
}: Props) {
  const ref = useRef<HTMLDivElement>(null);
  const [state, setState] = useState({ atStart: true, atEnd: true });

  useEffect(() => {
    const el = ref.current;
    if (!el) return;

    const update = () => {
      if (axis === "x") {
        const scrollable = el.scrollWidth - el.clientWidth > 1;
        const atStart = el.scrollLeft <= 1;
        const atEnd = el.scrollLeft + el.clientWidth >= el.scrollWidth - 1;
        setState({ atStart: !scrollable || atStart, atEnd: !scrollable || atEnd });
      } else {
        const scrollable = el.scrollHeight - el.clientHeight > 1;
        const atStart = el.scrollTop <= 1;
        const atEnd = el.scrollTop + el.clientHeight >= el.scrollHeight - 1;
        setState({ atStart: !scrollable || atStart, atEnd: !scrollable || atEnd });
      }
    };

    update();
    el.addEventListener("scroll", update, { passive: true });
    const ro = new ResizeObserver(update);
    ro.observe(el);
    return () => {
      el.removeEventListener("scroll", update);
      ro.disconnect();
    };
  }, [axis]);

  return (
    <div
      className={`scroll-shadow scroll-shadow-${axis}`}
      data-at-start={state.atStart ? "true" : "false"}
      data-at-end={state.atEnd ? "true" : "false"}
    >
      <div
        ref={ref}
        className={`scroll-shadow-inner ${className ?? ""} ${innerClassName ?? ""}`.trim()}
        style={innerStyle}
      >
        {children}
      </div>
    </div>
  );
}

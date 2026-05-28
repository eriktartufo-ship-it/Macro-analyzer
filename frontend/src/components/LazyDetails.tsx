import { useState, type ReactNode } from "react";

type Props = {
  summary: ReactNode;
  children: ReactNode;
  defaultOpen?: boolean;
  className?: string;
};

export function LazyDetails({ summary, children, defaultOpen = false, className = "card" }: Props) {
  const [hasOpened, setHasOpened] = useState(defaultOpen);

  return (
    <details
      className={className}
      open={defaultOpen}
      onToggle={(e) => {
        if ((e.target as HTMLDetailsElement).open) setHasOpened(true);
      }}
    >
      <summary style={{ cursor: "pointer", fontWeight: 600 }}>{summary}</summary>
      <div style={{ marginTop: 12, display: "flex", flexDirection: "column", gap: 12 }}>
        {hasOpened ? children : null}
      </div>
    </details>
  );
}

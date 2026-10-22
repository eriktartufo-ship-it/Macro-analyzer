import { SECTIONS, type Route, type Section } from "../lib/nav";

/** Schede di una sezione (segmented, convenzione della workspace per le sottopagine). */
export function SectionTabs({
  section,
  active,
  navigate,
}: {
  section: Section;
  active: string | null;
  navigate: (r: Route) => void;
}) {
  const def = SECTIONS.find((s) => s.id === section);
  if (!def || def.tabs.length === 0) return null;
  return (
    <div className="section-head">
      <h1 className="section-title">{def.label}</h1>
      <div className="segmented section-tabs" role="tablist" aria-label={`Schede di ${def.label}`}>
        {def.tabs.map((t) => (
          <button
            key={t.id}
            role="tab"
            aria-selected={active === t.id}
            className={active === t.id ? "active" : ""}
            onClick={() => navigate({ section, tab: t.id, asset: null })}
          >
            {t.label}
          </button>
        ))}
      </div>
    </div>
  );
}

/** Matches on the analysis server: open a finished one, follow one still being analysed. */
import { useCallback, useEffect, useState } from "react";
import type { Backend, MatchSummary } from "../platform/backend";
import { IconAlert, IconFilm, IconTrash } from "./Icons";

interface MatchLibraryProps {
  backend: Backend;
  onOpen: (video: string, name: string) => void;
}

const STATUS_TEXT: Record<MatchSummary["status"], string> = {
  uploading: "Upload unfinished",
  queued: "Queued",
  analyzing: "Analysing",
  complete: "Ready",
  error: "Failed",
  stopped: "Stopped",
};

export default function MatchLibrary({ backend, onOpen }: MatchLibraryProps) {
  const [matches, setMatches] = useState<MatchSummary[] | null>(null);
  const [error, setError] = useState<string | null>(null);

  const refresh = useCallback(() => {
    backend.listMatches?.()
      .then((m) => { setMatches(m); setError(null); })
      .catch((e) => setError(`Cannot reach the analysis server: ${String(e)}`));
  }, [backend]);

  useEffect(() => {
    refresh();
    // Analyses in progress move on their own.
    const id = window.setInterval(refresh, 5000);
    return () => window.clearInterval(id);
  }, [refresh]);

  return (
    <section className="library">
      <div className="library-head">
        <h2>Matches</h2>
        {matches && matches.length > 0 && <span className="match-dim">{matches.length}</span>}
      </div>
      {error && <div className="notice notice-bad"><IconAlert />{error}</div>}
      {matches?.length === 0 && (
        <div className="library-empty">
          <IconFilm size={22} />
          <p>No matches yet. Upload a video or record one and it will appear here.</p>
        </div>
      )}
      {matches && matches.length > 0 && (
        <div className="library-list">
          {matches.map((m) => (
            <div key={m.video} className="library-row">
              <div className="library-thumb"><IconFilm size={18} /></div>
              <div className="library-main">
                <div className="library-name" title={m.name}>{m.name}</div>
                <div className="match-dim">
                  {new Date(m.created).toLocaleString(undefined, { dateStyle: "medium", timeStyle: "short" })} · {(m.size / 1e6).toFixed(0)} MB
                  {m.status === "error" && m.message ? ` · ${m.message}` : ""}
                </div>
              </div>
              <span className={`status-pill ${m.status}`}>
                {STATUS_TEXT[m.status] ?? m.status}
                {m.status === "analyzing" && ` ${m.progress}%`}
              </span>
              <div className="library-actions">
                <button
                  className={`btn ${m.status === "complete" ? "btn-primary" : "btn-secondary"}`}
                  disabled={m.status === "uploading"}
                  onClick={() => onOpen(m.video, m.name)}
                >
                  {m.status === "complete" ? "Open" : m.status === "error" || m.status === "stopped" ? "Analyse again" : "Follow"}
                </button>
                <button
                  className="icon-btn icon-btn-danger"
                  title="Delete"
                  aria-label={`Delete ${m.name}`}
                  onClick={() => {
                    if (!confirm(`Delete "${m.name}" and its analysis from the server?`)) return;
                    backend.deleteMatch?.(m.video).then(refresh).catch((e) => setError(String(e)));
                  }}
                >
                  <IconTrash />
                </button>
              </div>
            </div>
          ))}
        </div>
      )}
    </section>
  );
}

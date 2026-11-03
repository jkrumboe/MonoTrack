/** Matches on the analysis server: open a finished one, follow one still being analysed. */
import { useCallback, useEffect, useState } from "react";
import type { Backend, MatchSummary } from "../platform/backend";

interface MatchLibraryProps {
  backend: Backend;
  onOpen: (video: string) => void;
  onClose: () => void;
}

const STATUS_TEXT: Record<MatchSummary["status"], string> = {
  uploading: "upload unfinished",
  queued: "waiting to be analysed",
  analyzing: "analysing",
  complete: "ready",
  error: "failed",
  stopped: "stopped",
};

export default function MatchLibrary({ backend, onOpen, onClose }: MatchLibraryProps) {
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
    <div className="library">
      <div className="library-head">
        <h2>Matches</h2>
        <button className="btn btn-secondary" onClick={onClose}>Back</button>
      </div>
      {error && <div className="record-error inline">{error}</div>}
      {matches?.length === 0 && <p className="match-dim">No matches yet. Upload a video or record one.</p>}
      {matches?.map((m) => (
        <div key={m.video} className="record-item">
          <div>
            <div>{m.name}</div>
            <div className="match-dim">
              {new Date(m.created).toLocaleString()} · {(m.size / 1e6).toFixed(0)} MB · {STATUS_TEXT[m.status] ?? m.status}
              {m.status === "analyzing" && ` ${m.progress}%`}
              {m.status === "error" && m.message ? ` — ${m.message}` : ""}
            </div>
          </div>
          <div className="record-item-actions">
            <button
              className="btn btn-primary"
              disabled={m.status === "uploading"}
              onClick={() => onOpen(m.video)}
            >
              {m.status === "complete" ? "Open" : m.status === "error" || m.status === "stopped" ? "Analyse again" : "Follow"}
            </button>
            <button
              className="btn btn-danger"
              onClick={() => {
                if (!confirm(`Delete "${m.name}" and its analysis from the server?`)) return;
                backend.deleteMatch?.(m.video).then(refresh).catch((e) => setError(String(e)));
              }}
            >
              Delete
            </button>
          </div>
        </div>
      ))}
    </div>
  );
}

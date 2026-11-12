/** The start screen: bring in a match, or pick one from the library. */
import type { ReactNode } from "react";
import { inTauri, isMobileDevice, saveSettings, type Backend, type Settings } from "../platform/backend";
import { touchColor } from "../utils/rallyEdits";
import { IconAlert, IconArrowRight, IconFolder, IconPlay, IconRecord, IconServer, IconUpload } from "./Icons";
import MatchLibrary from "./MatchLibrary";

interface HomeScreenProps {
  onServer: boolean;
  canRecord: boolean;
  backend: Backend | null;
  error: string | null;
  settings: Settings;
  onSettingsChange: (s: Settings) => void;
  onRecord: () => void;
  onSelectVideo: () => void;
  onDemo: () => void;
  onOpenMatch: (video: string, name: string) => void;
}

export default function HomeScreen(p: HomeScreenProps) {
  const { settings, onSettingsChange } = p;
  return (
    <div className="home">
      <div className="home-inner">
        <section className="home-hero">
          <div className="home-copy">
            <h1>Every rally, point and touch on one timeline</h1>
            <p>
              Record a match with a phone on a tripod, or bring a video. Serves, receptions, sets,
              attacks, blocks and digs are found for you, and the ball is rebuilt in 3D.
            </p>
          </div>
          <RallyIllustration />
        </section>

        {p.error && <div className="notice notice-bad"><IconAlert />{p.error}</div>}

        <section className="home-actions">
          {p.onServer && p.canRecord && (
            <ActionCard icon={<IconRecord size={20} />} title="Record a match" text="Use this device's camera, with a court guide and level." onClick={p.onRecord} />
          )}
          <ActionCard
            primary
            icon={p.onServer ? <IconUpload size={20} /> : <IconFolder size={20} />}
            title={p.onServer ? "Upload a video" : "Open a video"}
            text={p.onServer ? "Send a match to the analysis server." : "Analyse a match file on this computer."}
            onClick={p.onSelectVideo}
          />
          {/* Sample data comes from the local analyzer; a server only knows uploaded videos. */}
          {!p.onServer && (
            <ActionCard icon={<IconPlay size={18} />} title="Try the demo" text="Look around with sample data, no video needed." onClick={p.onDemo} />
          )}
        </section>

        {p.onServer && p.backend?.listMatches && <MatchLibrary backend={p.backend} onOpen={p.onOpenMatch} />}

        <details className="settings">
          <summary><IconServer />Analysis server</summary>
          <div className="settings-body">
            <label>
              Server address
              <input
                value={settings.serverUrl}
                placeholder="the server of this page"
                onChange={(e) => onSettingsChange({ ...settings, serverUrl: e.target.value.trim() })}
              />
            </label>
            <label>
              Access token
              <input
                type="password"
                value={settings.token}
                placeholder="none"
                onChange={(e) => onSettingsChange({ ...settings, token: e.target.value.trim() })}
              />
            </label>
            {inTauri() && !isMobileDevice() && (
              <label className="match-toggle">
                <input
                  type="checkbox"
                  checked={settings.useServerOnDesktop}
                  onChange={(e) => onSettingsChange({ ...settings, useServerOnDesktop: e.target.checked })}
                />
                Analyse on the server instead of this computer
              </label>
            )}
            <button className="btn btn-secondary" onClick={() => { saveSettings(settings); onSettingsChange({ ...settings }); }}>
              Save
            </button>
          </div>
        </details>
      </div>
    </div>
  );
}

function ActionCard({ icon, title, text, onClick, primary }: { icon: ReactNode; title: string; text: string; onClick: () => void; primary?: boolean }) {
  return (
    <button className={`action-card ${primary ? "primary" : ""}`} onClick={onClick}>
      <span className="action-card-icon">{icon}</span>
      <span className="action-card-title">{title}<IconArrowRight className="action-card-arrow" /></span>
      <span className="action-card-text">{text}</span>
    </button>
  );
}

/** A picture of what the product finds: one rally's touches over a court. Not data. */
function RallyIllustration() {
  const marks: { x: number; y: number; kind: string; label: string; anchor?: "start" | "end" }[] = [
    { x: 98, y: 262, kind: "serve", label: "Serve", anchor: "start" },
    { x: 352, y: 96, kind: "reception", label: "Reception", anchor: "start" },
    { x: 268, y: 76, kind: "set", label: "Set" },
    { x: 180, y: 58, kind: "attack", label: "Attack", anchor: "end" },
    { x: 312, y: 244, kind: "dig", label: "Dig", anchor: "start" },
  ];
  return (
    <svg className="home-illustration" viewBox="0 0 480 310" role="img" aria-label="A rally drawn over a volleyball court, with its touches marked">
      <polygon points="104,62 376,62 462,292 18,292" fill="var(--court-fill)" stroke="var(--court-line)" strokeWidth="1.5" />
      <line x1="84" y1="118" x2="396" y2="118" stroke="var(--court-line)" strokeWidth="1" opacity=".55" />
      <line x1="42" y1="228" x2="438" y2="228" stroke="var(--court-line)" strokeWidth="1" opacity=".55" />
      <line x1="66" y1="164" x2="414" y2="164" stroke="var(--court-line)" strokeWidth="1.5" />
      <polygon points="66,164 414,164 414,126 66,126" fill="var(--text-primary)" opacity=".07" />
      <line x1="66" y1="126" x2="414" y2="126" stroke="var(--text-primary)" strokeWidth="1.5" opacity=".7" />
      <line x1="66" y1="126" x2="66" y2="164" stroke="var(--text-secondary)" strokeWidth="1.5" />
      <line x1="414" y1="126" x2="414" y2="164" stroke="var(--text-secondary)" strokeWidth="1.5" />
      <path
        d="M98 262C150 60 300 30 352 96C330 58 296 56 268 76C240 34 205 34 180 58C210 120 270 190 312 244"
        fill="none"
        stroke="var(--ball)"
        strokeWidth="2"
        strokeLinecap="round"
        strokeDasharray="1 6"
      />
      {marks.map((m) => (
        <g key={m.kind}>
          <circle cx={m.x} cy={m.y} r="6" fill={touchColor(m.kind)} stroke="var(--bg-primary)" strokeWidth="2" />
          <text
            x={m.x + (m.anchor === "start" ? 12 : m.anchor === "end" ? -12 : 0)}
            y={m.anchor ? m.y + 4 : m.y + 22}
            textAnchor={m.anchor ?? "middle"}
            fill="var(--text-primary)"
            fontSize="12"
            fontWeight="500"
          >
            {m.label}
          </text>
        </g>
      ))}
    </svg>
  );
}

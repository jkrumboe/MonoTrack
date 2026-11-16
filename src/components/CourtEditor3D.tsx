/**
 * Edit the rally in 3D: the court in perspective, with the solved path and
 * its touches, to orbit and drag in (see RallyEditor3DView). Drag a touch to
 * where it really was; the solver holds it there and refits the flights
 * either side (see the pins in rallyEdits.ts).
 *
 * From one camera the depth along each line of sight is what the solver
 * cannot see, so the selected touch shows its line of sight (the ball is
 * somewhere on it); "snap to line of sight" makes a drag hold to it, and
 * Shift flips that for one drag. Grabbing the path anywhere else puts a new
 * touch there (the solver splits the flight at it).
 *
 * Each flight has a handle midway (a diamond): drag it up or down to make
 * the arc higher or flatter. Undo, redo and "reset flight" sit under the view.
 */
import { useEffect, useMemo, useRef, useState } from "react";
import type { FrameData } from "../hooks/useAnalysis";
import type { CalibrationResult } from "../utils/cameraSolver";
import { ballRay, touchColor, type EditedTouch } from "../utils/rallyEdits";
import type { RefitSegment } from "../utils/rallyRefit";
import RallyEditor3DView from "./RallyEditor3DView";

type V3 = [number, number, number];

/** Canvas height over width: the video's shape, so the two sit level side by side. */
const VIEW_ASPECT = 9 / 16;

interface Props {
  frames: FrameData[];
  positions: Map<number, V3>;
  touches: EditedTouch[];
  calibration: CalibrationResult | null;
  aspect: number;
  cur: number;
  /** Frame of the touch being edited, if any. */
  selected: number | null;
  onSelect: (frame: number) => void;
  onPin: (frame: number, pos: V3) => void;
  fps: number;
  /** The flights between touches, with where their arc handles sit. */
  segments: RefitSegment[];
  /** Set the height of the ball midway through the flight starting at a touch's frame. */
  onArc: (fromFrame: number, midY: number) => void;
  /** Put a flight back to the automatic result. */
  onResetSegment: (fromFrame: number, toFrame: number) => void;
  onUndo: () => void;
  onRedo: () => void;
  canUndo: boolean;
  canRedo: boolean;
  /** Draw attention to the editor (a problem's fix points here). */
  flash?: boolean;
}

export default function CourtEditor3D(p: Readonly<Props>) {
  const wrapRef = useRef<HTMLDivElement>(null);
  const [w, setW] = useState(0);
  /** Drags hold to the selected touch's line of sight (Shift flips it). */
  const [snap, setSnap] = useState(false);

  useEffect(() => {
    const el = wrapRef.current;
    if (!el) return;
    const ro = new ResizeObserver(() => setW(Math.max(0, Math.floor(el.clientWidth - 8))));
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  const h = Math.round(w * VIEW_ASPECT);
  const selTouch = p.touches.find((t) => t.frame === p.selected) ?? null;
  const ray = useMemo(
    () => (p.calibration && p.selected != null && p.aspect ? ballRay(p.frames, p.selected, p.calibration, p.aspect) : null),
    [p.calibration, p.selected, p.aspect, p.frames],
  );
  /** The flight that starts at the selected touch, or the one that ends at it for the last touch. */
  const selSegment = p.segments.find((x) => x.from === p.selected) ?? p.segments.find((x) => x.to === p.selected) ?? null;

  return (
    <div className={`court3d ${p.flash ? "flash" : ""}`} ref={wrapRef}>
      <div className="court3d-view">
        <div className="court3d-bar">
          <label className="court3d-snap" title="Hold dragged touches to the camera's line of sight through the ball. Shift flips it for one drag.">
            <input type="checkbox" checked={snap} onChange={(e) => setSnap(e.target.checked)} /> Snap to line of sight
          </label>
        </div>
        <RallyEditor3DView
          w={w} h={h} positions={p.positions} touches={p.touches} segments={p.segments} calibration={p.calibration}
          ray={ray} selected={p.selected} cur={p.cur} fps={p.fps} snap={snap}
          onSelect={p.onSelect} onPin={p.onPin} onArc={p.onArc}
        />
      </div>
      <div className="court3d-tools">
        <button className="inspect-btn" onClick={p.onUndo} disabled={!p.canUndo} title="Ctrl+Z">Undo</button>
        <button className="inspect-btn" onClick={p.onRedo} disabled={!p.canRedo} title="Ctrl+Shift+Z">Redo</button>
        {selSegment && (
          <button
            className="inspect-btn"
            onClick={() => p.onResetSegment(selSegment.from, selSegment.to)}
            disabled={!selSegment.rebuilt}
            title="Clear the arc and the 3D placements of this flight's two touches"
          >
            Reset flight to automatic
          </button>
        )}
        {selTouch && (
          <label className="court3d-height" title="Height of the ball at this touch (m). Enter to set it.">
            {selTouch.label} height
            <input
              key={`${selTouch.frame}:${selTouch.position[1]}`}
              type="number" step={0.1} min={0} max={30}
              defaultValue={selTouch.position[1].toFixed(2)}
              onKeyDown={(e) => { if (e.key === "Enter") e.currentTarget.blur(); }}
              onBlur={(e) => {
                const y = Number(e.currentTarget.value);
                const q = selTouch.position;
                if (Number.isFinite(y) && y >= 0 && Math.abs(y - q[1]) > 0.005) p.onPin(selTouch.frame, [q[0], Math.round(y * 100) / 100, q[2]]);
              }}
            />
            m
          </label>
        )}
        {selSegment?.warning && <span className="court3d-warn">This flight {selSegment.warning}.</span>}
      </div>
      <div className="court3d-help">
        {selTouch
          ? <>Drag the <b style={{ color: touchColor(selTouch.label) }}>{selTouch.label}</b> to where the ball was played.
              {ray ? ` The cyan line is the camera's line of sight: the ball is somewhere on it.${snap ? " Drags hold to it; Shift: place freely." : " Shift: hold to it."}` : " The ball is hidden at this touch."}</>
          : "Click a touch to select it, then drag it to where the ball was played. The flights either side follow."}
        {" "}Grab the path anywhere else to put a touch there.
        {" "}Drag a ball up or down for its height, or the ring under it along the floor (Alt flips) · drag empty space: orbit · right-drag: pan · wheel: zoom. Drag a diamond to make that flight higher or flatter.
      </div>
    </div>
  );
}

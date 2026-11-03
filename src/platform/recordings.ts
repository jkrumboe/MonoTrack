/**
 * Matches recorded on this device, kept until they are uploaded.
 *
 * A match is an hour or more of 1080p video - gigabytes - so it cannot sit in
 * memory. The recorder hands over a piece every few seconds and each piece is
 * written to IndexedDB straight away (browsers keep stored blobs on disk). A
 * crash, a dead battery or a closed tab loses only the last few seconds, and
 * the upload can wait for wifi.
 */

const DB_NAME = "vb-recordings";
const DB_VERSION = 1;

export interface Recording {
  id: string;
  name: string;
  mimeType: string;
  startedAt: string;
  durationS: number;
  bytes: number;
  chunks: number;
  /** Still being recorded (or the app stopped mid-recording). */
  recording: boolean;
  /** The server video it was (or is being) uploaded to. */
  uploadedTo?: string;
  uploadDone?: boolean;
}

let dbPromise: Promise<IDBDatabase> | null = null;

function db(): Promise<IDBDatabase> {
  dbPromise ??= new Promise((resolve, reject) => {
    const req = indexedDB.open(DB_NAME, DB_VERSION);
    req.onupgradeneeded = () => {
      const d = req.result;
      d.createObjectStore("recordings", { keyPath: "id" });
      d.createObjectStore("chunks"); // key: [recordingId, seq]
    };
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error);
  });
  return dbPromise;
}

function done(tx: IDBTransaction): Promise<void> {
  return new Promise((resolve, reject) => {
    tx.oncomplete = () => resolve();
    tx.onerror = () => reject(tx.error);
    tx.onabort = () => reject(tx.error);
  });
}

function result<T>(req: IDBRequest<T>): Promise<T> {
  return new Promise((resolve, reject) => {
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error);
  });
}

export async function createRecording(name: string, mimeType: string): Promise<Recording> {
  const rec: Recording = {
    id: `${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 8)}`,
    name, mimeType, startedAt: new Date().toISOString(), durationS: 0, bytes: 0, chunks: 0, recording: true,
  };
  const tx = (await db()).transaction("recordings", "readwrite");
  tx.objectStore("recordings").put(rec);
  await done(tx);
  return rec;
}

/** Store the next piece of a recording and update its totals. */
export async function addChunk(rec: Recording, blob: Blob, durationS: number): Promise<void> {
  const tx = (await db()).transaction(["recordings", "chunks"], "readwrite");
  tx.objectStore("chunks").put(blob, [rec.id, rec.chunks]);
  rec.chunks += 1;
  rec.bytes += blob.size;
  rec.durationS = durationS;
  tx.objectStore("recordings").put({ ...rec });
  await done(tx);
}

export async function updateRecording(id: string, change: Partial<Recording>): Promise<void> {
  const d = await db();
  const cur = await result(d.transaction("recordings").objectStore("recordings").get(id) as IDBRequest<Recording | undefined>);
  if (!cur) return;
  const tx = d.transaction("recordings", "readwrite");
  tx.objectStore("recordings").put({ ...cur, ...change });
  await done(tx);
}

export async function listRecordings(): Promise<Recording[]> {
  const all = await result((await db()).transaction("recordings").objectStore("recordings").getAll() as IDBRequest<Recording[]>);
  return all.sort((a, b) => b.startedAt.localeCompare(a.startedAt));
}

/** The whole recording as one file. The pieces stay on disk; the Blob only references them. */
export async function recordingBlob(rec: Recording): Promise<Blob> {
  const range = IDBKeyRange.bound([rec.id, 0], [rec.id, Number.MAX_SAFE_INTEGER]);
  const parts = await result((await db()).transaction("chunks").objectStore("chunks").getAll(range) as IDBRequest<Blob[]>);
  return new Blob(parts, { type: rec.mimeType });
}

export async function deleteRecording(id: string): Promise<void> {
  const tx = (await db()).transaction(["recordings", "chunks"], "readwrite");
  tx.objectStore("recordings").delete(id);
  tx.objectStore("chunks").delete(IDBKeyRange.bound([id, 0], [id, Number.MAX_SAFE_INTEGER]));
  await done(tx);
}

/** Ask the browser not to evict the recordings when the device is short of space. */
export async function requestPersistentStorage(): Promise<boolean> {
  try {
    return (await navigator.storage?.persist?.()) ?? false;
  } catch {
    return false;
  }
}

export async function storageEstimate(): Promise<{ usage: number; quota: number } | null> {
  try {
    const e = await navigator.storage?.estimate?.();
    return e ? { usage: e.usage ?? 0, quota: e.quota ?? 0 } : null;
  } catch {
    return null;
  }
}

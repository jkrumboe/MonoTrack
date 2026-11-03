use serde_json::Value;
use std::io::{BufRead, BufReader, Write};
use std::process::{Child, ChildStdin, Command, Stdio};
use std::sync::Mutex;
use tauri::{AppHandle, Emitter, Manager, State};

/// Persistent analysis process. Started with the app and kept alive between
/// analyses, so Python, torch and the detectors load once (~12 s) instead of
/// on every run. Stopping an analysis kills it and starts a fresh one.
#[derive(Default)]
struct Analyzer {
    child: Option<Child>,
    stdin: Option<ChildStdin>,
    /// An analysis was sent and has not completed or failed yet.
    busy: bool,
    /// Bumped per spawn, so a dying old process's reader cannot clear the
    /// state of the process that replaced it.
    generation: u64,
}

struct AnalyzerProcess(Mutex<Analyzer>);

/// Persistent court detector process — keeps model loaded in memory.
struct CourtDetectorProcess {
    child: Mutex<Option<Child>>,
    stdin: Mutex<Option<std::process::ChildStdin>>,
    stdout: Mutex<Option<BufReader<std::process::ChildStdout>>>,
}

/// Find the Python interpreter. Prefers a local .venv over the system Python
/// so that packages like ultralytics installed in the venv are available.
fn find_python() -> String {
    // Derive project root from CARGO_MANIFEST_DIR (src-tauri/) → go up one level
    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    let project_root = std::path::Path::new(manifest_dir).join("..");

    // Candidate venv executables (Windows first, then Unix-style)
    let venv_candidates = [
        project_root.join(".venv/Scripts/python.exe"),
        project_root.join(".venv/bin/python"),
        project_root.join("venv/Scripts/python.exe"),
        project_root.join("venv/bin/python"),
    ];

    for candidate in &venv_candidates {
        if let Ok(canonical) = candidate.canonicalize() {
            if canonical.exists() {
                eprintln!("[find_python] Using venv Python: {}", canonical.display());
                return canonical.to_string_lossy().to_string();
            }
        }
    }

    // Fall back to system Python
    for cmd in &["python", "python3"] {
        if Command::new(cmd)
            .arg("--version")
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status()
            .is_ok()
        {
            eprintln!("[find_python] Using system Python: {}", cmd);
            return cmd.to_string();
        }
    }
    "python".to_string()
}

/// Resolve the path to the analyzer.py script relative to the app.
fn analyzer_script_path(app: &AppHandle) -> String {
    // In dev, CARGO_MANIFEST_DIR points to src-tauri/, so go up one level
    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    let dev_path = std::path::Path::new(manifest_dir).join("../python/analyzer.py");
    if dev_path.exists() {
        return dev_path.to_string_lossy().to_string();
    }

    // In bundled mode, check the resource directory
    if let Ok(resource_dir) = app.path().resource_dir() {
        let bundled_path = resource_dir.join("python/analyzer.py");
        if bundled_path.exists() {
            return bundled_path.to_string_lossy().to_string();
        }
    }

    // Fallback
    "python/analyzer.py".to_string()
}

fn kill_analyzer(a: &mut Analyzer) {
    if let Some(mut child) = a.child.take() {
        let _ = child.kill();
    }
    a.stdin = None;
    a.busy = false;
}

/// Start the analysis process and the threads that forward its output.
fn spawn_analyzer(app: &AppHandle, a: &mut Analyzer) -> Result<(), String> {
    kill_analyzer(a);
    let python = find_python();
    let script = analyzer_script_path(app);

    let mut child = Command::new(&python)
        .arg(&script)
        .arg("--warm")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| format!("Failed to start Python analyzer: {}. Python: {}, Script: {}", e, python, script))?;

    let stdin = child.stdin.take().ok_or("Failed to get stdin")?;
    let stdout = child.stdout.take().ok_or("Failed to get stdout")?;
    let stderr = child.stderr.take().ok_or("Failed to get stderr")?;
    a.generation += 1;
    let generation = a.generation;
    a.child = Some(child);
    a.stdin = Some(stdin);

    std::thread::spawn(move || {
        let reader = BufReader::new(stderr);
        for line in reader.lines() {
            if let Ok(line) = line {
                eprintln!("[Python stderr] {}", line);
            }
        }
    });

    // Forward stdout lines to the frontend as events.
    let app = app.clone();
    std::thread::spawn(move || {
        let reader = BufReader::new(stdout);
        for line in reader.lines() {
            let line = match line {
                Ok(line) => line,
                Err(e) => {
                    eprintln!("[Python stdout read error] {}", e);
                    break;
                }
            };
            if line.trim().is_empty() {
                continue;
            }
            let json = match serde_json::from_str::<Value>(&line) {
                Ok(json) => json,
                Err(e) => {
                    eprintln!("[Python stdout parse error] {}: {}", e, line);
                    continue;
                }
            };
            let action = json.get("action").and_then(|a| a.as_str()).unwrap_or("unknown");
            let event = match action {
                "frame" => "analysis-frame",
                "progress" => "analysis-progress",
                "complete" => "analysis-complete",
                "error" => "analysis-error",
                "ready" => "analyzer-ready",
                _ => "analysis-message",
            };
            if action == "complete" || action == "error" {
                if let Ok(mut a) = app.state::<AnalyzerProcess>().0.lock() {
                    if a.generation == generation {
                        a.busy = false;
                    }
                }
            }
            let _ = app.emit(event, &json);
        }
        // The process ended. If it was mid-analysis (not stopped on purpose,
        // which clears `busy` first), report it.
        let was_busy = match app.state::<AnalyzerProcess>().0.lock() {
            Ok(mut a) if a.generation == generation => {
                let was_busy = a.busy;
                kill_analyzer(&mut a);
                was_busy
            }
            _ => false,
        };
        if was_busy {
            let _ = app.emit(
                "analysis-error",
                serde_json::json!({ "action": "error", "message": "Analyzer process exited unexpectedly" }),
            );
        }
    });
    Ok(())
}

fn send_analyze(a: &mut Analyzer, request: &str) -> Result<(), String> {
    let stdin = a.stdin.as_mut().ok_or("No stdin")?;
    stdin
        .write_all(request.as_bytes())
        .and_then(|_| stdin.flush())
        .map_err(|e| format!("Failed to write to Python stdin: {}", e))
}

#[tauri::command]
async fn start_analysis(
    app: AppHandle,
    state: State<'_, AnalyzerProcess>,
    video_path: String,
) -> Result<String, String> {
    let request = format!("{}
", serde_json::json!({ "action": "analyze", "video_path": video_path }));
    let mut a = state.0.lock().map_err(|e| format!("Lock error: {}", e))?;

    // Reuse the warm process unless it is gone or still busy with a previous
    // run (whose frames would otherwise mix into this one).
    let alive = a.child.as_mut().map_or(false, |c| matches!(c.try_wait(), Ok(None)));
    if !alive || a.busy {
        spawn_analyzer(&app, &mut a)?;
    }
    if send_analyze(&mut a, &request).is_err() {
        // The process died between runs: start it again once.
        spawn_analyzer(&app, &mut a)?;
        send_analyze(&mut a, &request)?;
    }
    a.busy = true;
    Ok("Analysis started".to_string())
}

#[tauri::command]
async fn detect_court(
    app: AppHandle,
    state: State<'_, CourtDetectorProcess>,
    video_path: String,
    frame_time: f64,
) -> Result<Value, String> {
    // Ensure the persistent Python process is running
    {
        let has_process = state.child.lock().map(|g| g.is_some()).unwrap_or(false);
        if !has_process {
            let python = find_python();
            let script = analyzer_script_path(&app);

            let mut child = Command::new(&python)
                .arg(&script)
                .stdin(Stdio::piped())
                .stdout(Stdio::piped())
                .stderr(Stdio::piped())
                .spawn()
                .map_err(|e| format!("Failed to start Python: {}. Python: {}, Script: {}", e, python, script))?;

            let stdin = child.stdin.take().ok_or("Failed to get stdin")?;
            let stdout = child.stdout.take().ok_or("Failed to get stdout")?;
            let stderr = child.stderr.take().ok_or("Failed to get stderr")?;

            // Log stderr in background
            std::thread::spawn(move || {
                let reader = BufReader::new(stderr);
                for line in reader.lines() {
                    if let Ok(line) = line {
                        eprintln!("[CourtDetector stderr] {}", line);
                    }
                }
            });

            // Wait for "ready" message
            let mut reader = BufReader::new(stdout);
            let mut ready_line = String::new();
            let _ = reader.read_line(&mut ready_line);

            if let Ok(mut guard) = state.child.lock() { *guard = Some(child); }
            if let Ok(mut guard) = state.stdin.lock() { *guard = Some(stdin); }
            if let Ok(mut guard) = state.stdout.lock() { *guard = Some(reader); }
        }
    }

    // Send detect_court request
    let request = serde_json::json!({
        "action": "detect_court",
        "video_path": video_path,
        "frame_time": frame_time
    });

    {
        let mut stdin_guard = state.stdin.lock().map_err(|e| format!("Lock error: {}", e))?;
        if let Some(ref mut stdin) = *stdin_guard {
            stdin
                .write_all(format!("{}\n", request).as_bytes())
                .map_err(|e| {
                    // Process died — clear state so next call restarts it
                    drop(stdin_guard);
                    if let Ok(mut g) = state.child.lock() { *g = None; }
                    if let Ok(mut g) = state.stdin.lock() { *g = None; }
                    if let Ok(mut g) = state.stdout.lock() { *g = None; }
                    format!("Write error (process may have died): {}", e)
                })?;
        } else {
            return Err("No stdin available".to_string());
        }
    }

    // Read response
    {
        let mut stdout_guard = state.stdout.lock().map_err(|e| format!("Lock error: {}", e))?;
        if let Some(ref mut reader) = *stdout_guard {
            loop {
                let mut line = String::new();
                match reader.read_line(&mut line) {
                    Ok(0) => {
                        // EOF — process died, clear state
                        drop(stdout_guard);
                        if let Ok(mut g) = state.child.lock() { *g = None; }
                        if let Ok(mut g) = state.stdin.lock() { *g = None; }
                        if let Ok(mut g) = state.stdout.lock() { *g = None; }
                        return Err("Court detector process ended unexpectedly".to_string());
                    }
                    Ok(_) => {
                        let line = line.trim().to_string();
                        if line.is_empty() { continue; }
                        if let Ok(json) = serde_json::from_str::<Value>(&line) {
                            let action = json.get("action").and_then(|a| a.as_str()).unwrap_or("");
                            match action {
                                "court_detected" | "court_failed" | "error" => {
                                    return Ok(json);
                                }
                                _ => continue, // skip "ready" etc.
                            }
                        }
                    }
                    Err(e) => {
                        drop(stdout_guard);
                        if let Ok(mut g) = state.child.lock() { *g = None; }
                        if let Ok(mut g) = state.stdin.lock() { *g = None; }
                        if let Ok(mut g) = state.stdout.lock() { *g = None; }
                        return Err(format!("Read error: {}", e));
                    }
                }
            }
        } else {
            return Err("No stdout available".to_string());
        }
    }
}

/// Store a hand-placed court so later clips of the same match calibrate
/// themselves (python/court_store.py). Runs the analyzer once and waits.
#[tauri::command]
async fn save_court(app: AppHandle, video_path: String, frame: u32, corners: Vec<[f64; 2]>) -> Result<String, String> {
    let request = serde_json::json!({ "video_path": video_path, "frame": frame, "corners": corners });
    let output = Command::new(find_python())
        .arg(analyzer_script_path(&app))
        .arg("--save-court")
        .arg(request.to_string())
        .output()
        .map_err(|e| format!("Failed to start Python: {}", e))?;
    if !output.status.success() {
        return Err(String::from_utf8_lossy(&output.stderr).to_string());
    }
    Ok(String::from_utf8_lossy(&output.stdout).trim().to_string())
}

#[tauri::command]
async fn stop_analysis(app: AppHandle, state: State<'_, AnalyzerProcess>) -> Result<String, String> {
    // Killing is the only way to interrupt a run; a fresh process starts
    // straight away so the next analysis finds the detectors loaded.
    if let Ok(mut a) = state.0.lock() {
        spawn_analyzer(&app, &mut a)?;
    }
    Ok("Analysis stopped".to_string())
}

/// Write `content` to the file at `path`. Used by the frontend export helpers
/// because Tauri's WebView does not support the `<a download>` blob-URL trick.
#[tauri::command]
fn write_text_file(path: String, content: String) -> Result<(), String> {
    std::fs::write(&path, content).map_err(|e| format!("Failed to write {}: {}", path, e))
}

/// Where a video's hand corrections (Inspect & Edit) are kept: next to the
/// saved courts in ~/.tennis-3d (TENNIS3D_EDITS overrides), named by the
/// video's file name and size, so they are found again after the video is
/// moved or re-analysed.
fn edits_file(video_path: &str) -> Result<std::path::PathBuf, String> {
    let size = std::fs::metadata(video_path)
        .map_err(|e| format!("Cannot read {}: {}", video_path, e))?
        .len();
    let stem: String = std::path::Path::new(video_path)
        .file_stem()
        .map(|s| s.to_string_lossy().to_string())
        .unwrap_or_default()
        .chars()
        .map(|c| if c.is_ascii_alphanumeric() || c == '-' || c == '_' { c } else { '_' })
        .take(60)
        .collect();
    let dir = match std::env::var("TENNIS3D_EDITS") {
        Ok(d) if !d.is_empty() => std::path::PathBuf::from(d),
        _ => {
            let home = std::env::var("USERPROFILE")
                .or_else(|_| std::env::var("HOME"))
                .map_err(|_| "No home directory".to_string())?;
            std::path::Path::new(&home).join(".tennis-3d").join("edits")
        }
    };
    Ok(dir.join(format!("{}-{}.json", stem, size)))
}

/// The saved corrections for a video, or None when it has none yet.
#[tauri::command]
fn load_edits(video_path: String) -> Result<Option<String>, String> {
    let file = edits_file(&video_path)?;
    if !file.exists() {
        return Ok(None);
    }
    std::fs::read_to_string(&file)
        .map(Some)
        .map_err(|e| format!("Failed to read {}: {}", file.display(), e))
}

/// Save a video's corrections; returns the file they went to.
#[tauri::command]
fn save_edits(video_path: String, content: String) -> Result<String, String> {
    let file = edits_file(&video_path)?;
    if let Some(dir) = file.parent() {
        std::fs::create_dir_all(dir).map_err(|e| format!("Failed to create {}: {}", dir.display(), e))?;
    }
    // Write then rename, so a crash mid-save cannot leave half a file.
    let tmp = file.with_extension("json.tmp");
    std::fs::write(&tmp, content)
        .and_then(|_| std::fs::rename(&tmp, &file))
        .map_err(|e| format!("Failed to write {}: {}", file.display(), e))?;
    Ok(file.to_string_lossy().to_string())
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_opener::init())
        .plugin(tauri_plugin_shell::init())
        .plugin(tauri_plugin_dialog::init())
        .manage(AnalyzerProcess(Mutex::new(Analyzer::default())))
        .manage(CourtDetectorProcess {
            child: Mutex::new(None),
            stdin: Mutex::new(None),
            stdout: Mutex::new(None),
        })
        .setup(|app| {
            // Load the detectors while the user is still picking a video.
            let handle = app.handle().clone();
            if let Ok(mut a) = app.state::<AnalyzerProcess>().0.lock() {
                if let Err(e) = spawn_analyzer(&handle, &mut a) {
                    eprintln!("[analyzer] {}", e);
                }
            }
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![start_analysis, stop_analysis, detect_court, save_court, write_text_file, load_edits, save_edits])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}

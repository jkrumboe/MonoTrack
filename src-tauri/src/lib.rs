use serde_json::Value;
use std::io::{BufRead, BufReader, Write};
use std::process::{Child, Command, Stdio};
use std::sync::Mutex;
use tauri::{AppHandle, Emitter, Manager, State};

struct AnalyzerProcess(Mutex<Option<Child>>);

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

#[tauri::command]
async fn start_analysis(
    app: AppHandle,
    state: State<'_, AnalyzerProcess>,
    video_path: String,
) -> Result<String, String> {
    // Kill any existing analyzer process
    if let Ok(mut guard) = state.0.lock() {
        if let Some(ref mut child) = *guard {
            let _ = child.kill();
        }
        *guard = None;
    }

    let python = find_python();
    let script = analyzer_script_path(&app);

    let mut child = Command::new(&python)
        .arg(&script)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| format!("Failed to start Python analyzer: {}. Python: {}, Script: {}", e, python, script))?;

    let mut stdin = child.stdin.take().ok_or("Failed to get stdin")?;
    let stdout = child.stdout.take().ok_or("Failed to get stdout")?;
    let stderr = child.stderr.take().ok_or("Failed to get stderr")?;

    // Store the child process
    if let Ok(mut guard) = state.0.lock() {
        *guard = Some(child);
    }

    let app_handle = app.clone();

    // Spawn a thread to read stderr and log it
    std::thread::spawn(move || {
        let reader = BufReader::new(stderr);
        for line in reader.lines() {
            if let Ok(line) = line {
                eprintln!("[Python stderr] {}", line);
            }
        }
    });

    // Spawn a thread to read stdout and emit events to the frontend
    let app_for_stdout = app_handle.clone();
    std::thread::spawn(move || {
        let reader = BufReader::new(stdout);
        for line in reader.lines() {
            match line {
                Ok(line) => {
                    if line.trim().is_empty() {
                        continue;
                    }
                    match serde_json::from_str::<Value>(&line) {
                        Ok(json) => {
                            let action = json
                                .get("action")
                                .and_then(|a| a.as_str())
                                .unwrap_or("unknown");

                            match action {
                                "frame" => {
                                    let _ = app_for_stdout.emit("analysis-frame", &json);
                                }
                                "progress" => {
                                    let _ = app_for_stdout.emit("analysis-progress", &json);
                                }
                                "complete" => {
                                    let _ = app_for_stdout.emit("analysis-complete", &json);
                                }
                                "error" => {
                                    let _ = app_for_stdout.emit("analysis-error", &json);
                                }
                                "ready" => {
                                    let _ = app_for_stdout.emit("analyzer-ready", &json);
                                }
                                _ => {
                                    let _ = app_for_stdout.emit("analysis-message", &json);
                                }
                            }
                        }
                        Err(e) => {
                            eprintln!("[Python stdout parse error] {}: {}", e, line);
                        }
                    }
                }
                Err(e) => {
                    eprintln!("[Python stdout read error] {}", e);
                    break;
                }
            }
        }
    });

    // Send the analyze command to the Python process
    let request = serde_json::json!({
        "action": "analyze",
        "video_path": video_path
    });

    let msg = format!("{}\n", request);
    stdin
        .write_all(msg.as_bytes())
        .map_err(|e| format!("Failed to write to Python stdin: {}", e))?;

    // We intentionally drop stdin here so the Python process knows
    // no more input is coming for this request. For a long-lived process,
    // you'd keep stdin open. For simplicity, we close it after sending.
    drop(stdin);

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

#[tauri::command]
async fn stop_analysis(state: State<'_, AnalyzerProcess>) -> Result<String, String> {
    if let Ok(mut guard) = state.0.lock() {
        if let Some(ref mut child) = *guard {
            let _ = child.kill();
        }
        *guard = None;
    }
    Ok("Analysis stopped".to_string())
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_opener::init())
        .plugin(tauri_plugin_shell::init())
        .plugin(tauri_plugin_dialog::init())
        .manage(AnalyzerProcess(Mutex::new(None)))
        .manage(CourtDetectorProcess {
            child: Mutex::new(None),
            stdin: Mutex::new(None),
            stdout: Mutex::new(None),
        })
        .invoke_handler(tauri::generate_handler![start_analysis, stop_analysis, detect_court])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}

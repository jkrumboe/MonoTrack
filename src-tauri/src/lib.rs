use serde_json::Value;
use std::io::{BufRead, BufReader, Write};
use std::process::{Child, Command, Stdio};
use std::sync::Mutex;
use tauri::{AppHandle, Emitter, Manager, State};

struct AnalyzerProcess(Mutex<Option<Child>>);

/// Find the Python interpreter. Tries "python" then "python3".
fn find_python() -> String {
    for cmd in &["python", "python3"] {
        if Command::new(cmd)
            .arg("--version")
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status()
            .is_ok()
        {
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
        return dev_path.canonicalize().unwrap_or(dev_path).to_string_lossy().to_string();
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
    video_path: String,
    frame_time: f64,
) -> Result<Value, String> {
    let python = find_python();
    let script = analyzer_script_path(&app);

    let mut child = Command::new(&python)
        .arg(&script)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| format!("Failed to start Python: {}. Python: {}, Script: {}", e, python, script))?;

    let mut stdin = child.stdin.take().ok_or("Failed to get stdin")?;
    let stdout = child.stdout.take().ok_or("Failed to get stdout")?;

    // Send detect_court command then quit
    let request = serde_json::json!({
        "action": "detect_court",
        "video_path": video_path,
        "frame_time": frame_time
    });
    stdin
        .write_all(format!("{}\n", request).as_bytes())
        .map_err(|e| format!("Write error: {}", e))?;
    stdin
        .write_all(b"{\"action\":\"quit\"}\n")
        .map_err(|e| format!("Write error: {}", e))?;
    drop(stdin);

    // Read stdout for the response
    let reader = BufReader::new(stdout);
    let mut result: Option<Value> = None;

    for line in reader.lines() {
        let line = line.map_err(|e| format!("Read error: {}", e))?;
        if line.trim().is_empty() {
            continue;
        }
        if let Ok(json) = serde_json::from_str::<Value>(&line) {
            let action = json.get("action").and_then(|a| a.as_str()).unwrap_or("");
            match action {
                "court_detected" | "court_failed" | "error" => {
                    result = Some(json);
                    break;
                }
                _ => {} // skip "ready" etc.
            }
        }
    }

    let _ = child.wait();

    result.ok_or_else(|| "No response from court detection".to_string())
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
        .invoke_handler(tauri::generate_handler![start_analysis, stop_analysis, detect_court])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}

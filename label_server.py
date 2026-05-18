"""
label_server.py
===============
Sleek, high-performance web labeling tool for CAPTCHAs.
Supports two modes automatically:
1. CSV Mode: If a metadata.csv exists in the target directory, it reads and updates labels inside the CSV.
2. Renaming Mode: If no metadata.csv exists (e.g. in a raw maps folder), it displays unlabeled map_*.png files
   and renames them in-place to map_LABEL.png upon saving!

Usage:
    python label_server.py [--dir DIRECTORY] [--port PORT]
"""

import os
import sys
import re
import csv
import json
import argparse
import mimetypes
import shutil
from pathlib import Path
from http.server import HTTPServer, SimpleHTTPRequestHandler
from urllib.parse import parse_qs, urlparse

# Default directories to check
DEFAULT_MAPS_DIR = Path(r"C:\Users\Administrator\Desktop\Server-1\maps")
DEFAULT_DATA_DIR = Path(__file__).parent / "data"

class LabelHandler(SimpleHTTPRequestHandler):
    """HTTP handler for CAPTCHA Labeling Tool."""

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path

        if path == "/" or path == "/index.html":
            self._send_html()
        elif path == "/api/config":
            self._send_config()
        elif path == "/api/metadata":
            self._send_metadata()
        elif path.startswith("/images/"):
            self._send_image(path)
        else:
            self.send_error(404)

    def do_POST(self) -> None:
        parsed = urlparse(self.path)

        if parsed.path == "/api/save":
            self._save_label()
        else:
            self.send_error(404)

    def _send_html(self) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.end_headers()
        self.wfile.write(HTML_PAGE.encode("utf-8"))

    def _send_config(self) -> None:
        mode = "CSV (metadata.csv)" if self.server.csv_mode else "Renaming (map_LABEL.png)"
        data = {
            "directory": str(self.server.data_dir),
            "csv_mode": self.server.csv_mode,
            "mode_name": mode
        }
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(data).encode("utf-8"))

    def _send_metadata(self) -> None:
        rows = self.server.read_data()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(rows, ensure_ascii=False).encode("utf-8"))

    def _send_image(self, path: str) -> None:
        filename = path.replace("/images/", "")
        
        # Security check: prevent directory traversal
        filepath = (self.server.data_dir / filename).resolve()
        if not filepath.is_relative_to(self.server.data_dir.resolve()) or not filepath.exists():
            self.send_error(404)
            return

        mime_type = mimetypes.guess_type(str(filepath))[0] or "image/png"
        self.send_response(200)
        self.send_header("Content-Type", mime_type)
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        with open(filepath, "rb") as f:
            self.wfile.write(f.read())

    def _save_label(self) -> None:
        content_length = int(self.headers["Content-Length"])
        body = self.rfile.read(content_length)
        data = json.loads(body)

        filename = data.get("filename")
        text = data.get("text", "").strip().upper()
        index = data.get("index")

        if not filename or len(text) != 5 or not text.isalnum():
            self.send_error(400, "Invalid label text or filename")
            return

        success = self.server.save_label(index, filename, text)
        if success:
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"ok": True}).encode("utf-8"))
        else:
            self.send_error(500, "Failed to save label")

    def log_message(self, format, *args) -> None:
        """Only log API calls to minimize terminal noise."""
        if "/api/" in str(args[0]):
            super().log_message(format, *args)


class LabelingServer(HTTPServer):
    """Custom HTTP Server holding labeling state and mode logic."""

    def __init__(self, server_address, RequestHandlerClass, data_dir: Path):
        self.data_dir = Path(data_dir).resolve()
        self.metadata_path = self.data_dir / "metadata.csv"
        self.csv_mode = self.metadata_path.exists()
        
        print(f"[DIR] Target Directory: {self.data_dir}")
        if self.csv_mode:
            print(f"[MODE] CSV mode (updating {self.metadata_path.name})")
        else:
            print(f"[MODE] Renaming mode (renaming map_*.png directly)")

        super().__init__(server_address, RequestHandlerClass)

    def read_data(self) -> list[dict]:
        """Read data depending on the active mode."""
        if self.csv_mode:
            # CSV Mode: Read from metadata.csv
            rows = []
            if self.metadata_path.exists():
                with open(self.metadata_path, "r", encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    for idx, row in enumerate(reader):
                        rows.append({
                            "index": idx,
                            "filename": row["filename"],
                            "text": row.get("text", "")
                        })
            return rows
        else:
            # Renaming Mode: Scan files dynamically
            rows = []
            files = sorted(self.data_dir.glob("map_*.png"))
            
            # Pattern to match: map_00000.png or map_LABEL.png
            # If it has a 5-letter uppercase alphanumeric label, we consider it already labeled!
            label_pattern = re.compile(r"^map_([A-Z0-9]{5})\.png$", re.IGNORECASE)
            number_pattern = re.compile(r"^map_\d+\.png$")

            for idx, filepath in enumerate(files):
                filename = filepath.name
                match_label = label_pattern.match(filename)
                
                if match_label:
                    text = match_label.group(1).upper()
                else:
                    text = "" # Unlabeled

                rows.append({
                    "index": idx,
                    "filename": filename,
                    "text": text
                })
            return rows

    def save_label(self, index: int, filename: str, text: str) -> bool:
        """Save the label according to the active mode."""
        if self.csv_mode:
            # CSV Mode: read, mutate, and write back metadata.csv
            rows = []
            with open(self.metadata_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                fieldnames = reader.fieldnames
                for r in reader:
                    rows.append(r)

            if 0 <= index < len(rows) and rows[index]["filename"] == filename:
                rows[index]["text"] = text
                with open(self.metadata_path, "w", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerows(rows)
                return True
            return False
        else:
            # Renaming Mode: rename the file map_*.png -> map_LABEL.png
            src_path = self.data_dir / filename
            if not src_path.exists():
                return False

            # Create clean labeled filename
            dest_filename = f"map_{text}.png"
            dest_path = self.data_dir / dest_filename

            # If it already exists, handle duplicate by adding a safe suffix
            if dest_path.exists() and dest_filename != filename:
                counter = 1
                while True:
                    dest_filename = f"map_{text}_{counter}.png"
                    dest_path = self.data_dir / dest_filename
                    if not dest_path.exists():
                        break
                    counter += 1

            try:
                shutil.move(str(src_path), str(dest_path))
                print(f"[RENAME] {filename} -> {dest_filename}")
                return True
            except Exception as e:
                print(f"[ERROR] Error renaming {filename}: {e}")
                return False


HTML_PAGE = """<!DOCTYPE html>
<html lang="vi">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>CAPTCHA Labeling Hub</title>
    <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&family=Space+Grotesk:wght@400;700&display=swap" rel="stylesheet">
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        :root {
            --bg-color: #0c0817;
            --glass-bg: rgba(255, 255, 255, 0.03);
            --glass-border: rgba(255, 255, 255, 0.08);
            --primary: #00f0ff;
            --primary-glow: rgba(0, 240, 255, 0.3);
            --accent: #9d4edd;
            --success: #00ff88;
            --success-glow: rgba(0, 255, 136, 0.2);
            --text: #f3effa;
            --text-muted: #8c83a2;
        }

        body {
            font-family: 'Outfit', sans-serif;
            background: radial-gradient(circle at 50% 0%, #1e113a 0%, var(--bg-color) 70%);
            color: var(--text);
            min-height: 100vh;
            display: flex;
            flex-direction: column;
            align-items: center;
            padding: 40px 20px;
            overflow-x: hidden;
        }

        /* Ambient glowing circles */
        .ambient-glow {
            position: absolute;
            width: 600px;
            height: 600px;
            background: radial-gradient(circle, rgba(157, 78, 221, 0.15) 0%, rgba(0,0,0,0) 70%);
            top: -200px;
            z-index: -1;
            filter: blur(50px);
            pointer-events: none;
        }

        header {
            text-align: center;
            margin-bottom: 30px;
        }

        h1 {
            font-family: 'Space Grotesk', sans-serif;
            font-size: 2.8rem;
            font-weight: 800;
            background: linear-gradient(135deg, #00f0ff, #c77dff);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            letter-spacing: -1px;
            margin-bottom: 8px;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 12px;
        }

        .stats-container {
            backdrop-filter: blur(12px);
            background: var(--glass-bg);
            border: 1px solid var(--glass-border);
            padding: 10px 24px;
            border-radius: 50px;
            font-size: 0.95rem;
            color: var(--text-muted);
            display: flex;
            gap: 20px;
            align-items: center;
            box-shadow: 0 4px 30px rgba(0, 0, 0, 0.4);
            margin-bottom: 25px;
        }

        .stats-container span {
            color: var(--primary);
            font-weight: 600;
        }

        .stats-divider {
            width: 1px;
            height: 16px;
            background: var(--glass-border);
        }

        .progress-container {
            width: 500px;
            height: 6px;
            background: rgba(255, 255, 255, 0.05);
            border-radius: 10px;
            margin-bottom: 30px;
            overflow: hidden;
            border: 1px solid rgba(255, 255, 255, 0.03);
        }

        .progress-bar {
            height: 100%;
            width: 0%;
            background: linear-gradient(90deg, var(--primary), var(--accent));
            border-radius: 10px;
            box-shadow: 0 0 10px var(--primary-glow);
            transition: width 0.4s cubic-bezier(0.1, 0.8, 0.3, 1);
        }

        .mode-badge {
            font-size: 0.75rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 1px;
            padding: 4px 10px;
            border-radius: 6px;
            background: rgba(0, 240, 255, 0.1);
            border: 1px solid var(--primary);
            color: var(--primary);
            margin-top: 5px;
            display: inline-block;
        }

        .filter-buttons {
            display: flex;
            gap: 12px;
            margin-bottom: 30px;
            background: rgba(255, 255, 255, 0.02);
            padding: 6px;
            border-radius: 14px;
            border: 1px solid var(--glass-border);
        }

        .filter-buttons button {
            background: transparent;
            border: none;
            color: var(--text-muted);
            padding: 8px 18px;
            font-size: 0.9rem;
            font-weight: 600;
            border-radius: 10px;
            cursor: pointer;
            transition: all 0.3s ease;
        }

        .filter-buttons button.active {
            background: var(--primary);
            color: var(--bg-color);
            box-shadow: 0 0 15px var(--primary-glow);
        }

        .workspace {
            display: flex;
            flex-direction: column;
            align-items: center;
            gap: 25px;
            width: 500px;
        }

        .card {
            width: 100%;
            backdrop-filter: blur(20px);
            background: rgba(255, 255, 255, 0.02);
            border: 1px solid var(--glass-border);
            border-radius: 24px;
            padding: 30px;
            box-shadow: 0 20px 50px rgba(0, 0, 0, 0.5);
            display: flex;
            flex-direction: column;
            align-items: center;
            gap: 20px;
            position: relative;
        }

        .card::before {
            content: '';
            position: absolute;
            top: 0; left: 0; right: 0; bottom: 0;
            border-radius: 24px;
            padding: 1px;
            background: linear-gradient(180deg, rgba(255,255,255,0.08), rgba(255,255,255,0));
            -webkit-mask: linear-gradient(#fff 0 0) content-box, linear-gradient(#fff 0 0);
            -webkit-mask-composite: xor;
            mask-composite: exclude;
            pointer-events: none;
        }

        .filename-banner {
            width: 100%;
            display: flex;
            justify-content: space-between;
            align-items: center;
            font-size: 0.85rem;
            color: var(--text-muted);
            border-bottom: 1px solid var(--glass-border);
            padding-bottom: 15px;
        }

        .filename-text {
            font-family: 'Space Grotesk', sans-serif;
            font-weight: 700;
            color: var(--text);
            letter-spacing: 0.5px;
        }

        .jump-box {
            display: flex;
            gap: 6px;
            align-items: center;
        }

        .jump-box input {
            width: 60px;
            background: rgba(0, 0, 0, 0.3);
            border: 1px solid var(--glass-border);
            color: var(--text);
            padding: 4px 8px;
            border-radius: 6px;
            outline: none;
            font-size: 0.8rem;
            text-align: center;
            -moz-appearance: textfield;
        }
        .jump-box input::-webkit-outer-spin-button,
        .jump-box input::-webkit-inner-spin-button {
            -webkit-appearance: none;
            margin: 0;
        }

        .jump-box button {
            background: var(--accent);
            color: var(--text);
            border: none;
            padding: 4px 10px;
            font-size: 0.75rem;
            font-weight: 600;
            border-radius: 6px;
            cursor: pointer;
            transition: all 0.2s;
        }
        .jump-box button:hover {
            filter: brightness(1.2);
        }

        .image-viewer {
            width: 100%;
            height: 180px;
            background: #110d24;
            border-radius: 16px;
            display: flex;
            justify-content: center;
            align-items: center;
            border: 1px solid rgba(255, 255, 255, 0.03);
            overflow: hidden;
            box-shadow: inset 0 0 20px rgba(0,0,0,0.8);
        }

        .image-viewer img {
            width: 256px;
            height: 128px;
            image-rendering: pixelated;
            object-fit: contain;
            border-radius: 4px;
            filter: drop-shadow(0 0 10px rgba(0, 240, 255, 0.2));
            transition: transform 0.2s ease;
        }
        
        .image-viewer img:hover {
            transform: scale(1.05);
        }

        .input-group {
            width: 100%;
            display: flex;
            gap: 12px;
            margin-top: 10px;
        }

        .input-group input {
            flex: 1;
            font-family: 'Space Grotesk', sans-serif;
            font-size: 1.8rem;
            font-weight: 700;
            background: rgba(0, 0, 0, 0.4);
            border: 2px solid var(--glass-border);
            color: var(--text);
            border-radius: 14px;
            padding: 12px;
            text-align: center;
            letter-spacing: 6px;
            text-transform: uppercase;
            outline: none;
            transition: all 0.3s;
        }

        .input-group input:focus {
            border-color: var(--primary);
            box-shadow: 0 0 20px rgba(0, 240, 255, 0.15);
            background: rgba(0, 0, 0, 0.6);
        }

        .btn-submit {
            background: linear-gradient(135deg, var(--primary), var(--accent));
            color: var(--bg-color);
            border: none;
            border-radius: 14px;
            font-size: 1.1rem;
            font-weight: 700;
            padding: 0 28px;
            cursor: pointer;
            transition: all 0.3s ease;
            box-shadow: 0 4px 20px rgba(0, 240, 255, 0.2);
        }

        .btn-submit:hover {
            transform: translateY(-2px);
            box-shadow: 0 8px 25px rgba(0, 240, 255, 0.4);
            filter: brightness(1.1);
        }

        .navigation {
            width: 100%;
            display: flex;
            gap: 15px;
        }

        .navigation button {
            flex: 1;
            background: rgba(255, 255, 255, 0.03);
            border: 1px solid var(--glass-border);
            color: var(--text-muted);
            font-size: 0.95rem;
            font-weight: 600;
            padding: 12px;
            border-radius: 12px;
            cursor: pointer;
            transition: all 0.2s;
            display: flex;
            justify-content: center;
            align-items: center;
            gap: 8px;
        }

        .navigation button:hover {
            background: rgba(255, 255, 255, 0.08);
            color: var(--text);
        }

        .keyboard-shortcuts {
            display: flex;
            gap: 15px;
            justify-content: center;
            font-size: 0.75rem;
            color: var(--text-muted);
            margin-top: 10px;
        }

        .keyboard-shortcuts kbd {
            background: rgba(0, 0, 0, 0.4);
            border: 1px solid var(--glass-border);
            color: var(--primary);
            padding: 2px 6px;
            border-radius: 4px;
            font-family: inherit;
        }

        /* Floating Toast Message */
        .toast {
            position: fixed;
            bottom: 40px;
            background: rgba(0, 255, 136, 0.15);
            backdrop-filter: blur(10px);
            border: 1px solid var(--success);
            color: var(--success);
            padding: 12px 24px;
            border-radius: 50px;
            font-weight: 600;
            font-size: 0.95rem;
            opacity: 0;
            transform: translateY(20px);
            transition: all 0.3s cubic-bezier(0.1, 0.8, 0.3, 1);
            z-index: 1000;
            box-shadow: 0 10px 30px var(--success-glow);
            display: flex;
            align-items: center;
            gap: 8px;
        }

        .toast.show {
            opacity: 1;
            transform: translateY(0);
        }

        .info-panel {
            font-size: 0.8rem;
            color: var(--text-muted);
            text-align: center;
            margin-top: 15px;
            line-height: 1.4;
        }
    </style>
</head>
<body>
    <div class="ambient-glow"></div>

    <header>
        <h1>🏷️ CAPTCHA Labeling Hub</h1>
        <div id="mode-badge" class="mode-badge">Đang tải cấu hình...</div>
    </header>

    <div class="stats-container">
        <div>Đã gán nhãn: <span id="labeled-count">0</span> / <span id="total-count">0</span></div>
        <div class="stats-divider"></div>
        <div>Còn lại: <span id="remaining-count">0</span></div>
    </div>

    <div class="progress-container">
        <div class="progress-bar" id="progress-bar"></div>
    </div>

    <div class="filter-buttons">
        <button id="btn-unlabeled" class="active" onclick="setFilter('unlabeled')">Chưa gán nhãn</button>
        <button id="btn-all" onclick="setFilter('all')">Tất cả</button>
        <button id="btn-labeled" onclick="setFilter('labeled')">Đã gán nhãn</button>
    </div>

    <div class="workspace">
        <div class="card">
            <div class="filename-banner">
                <span id="filename-label" class="filename-text">map_00000.png</span>
                <div class="jump-box">
                    <input type="number" id="goto-input" min="1" placeholder="STT...">
                    <button onclick="gotoImage()">Đi</button>
                </div>
            </div>

            <div class="image-viewer">
                <img id="captcha-img" src="" alt="Vui lòng gán nhãn">
            </div>

            <div class="input-group">
                <input type="text" id="label-input" placeholder="NHẬP 5 KÝ TỰ" maxlength="5" autofocus autocomplete="off" oninput="this.value = this.value.toUpperCase().replace(/[^A-Z0-9]/g, '')">
                <button class="btn-submit" onclick="saveLabel()">LƯU</button>
            </div>

            <div class="navigation">
                <button onclick="prevImage()">← Trước</button>
                <button onclick="nextImage()">Bỏ qua →</button>
            </div>
        </div>

        <div class="keyboard-shortcuts">
            <div><kbd>Enter</kbd> Lưu & Tiếp</div>
            <div><kbd>←</kbd> Ảnh Trước</div>
            <div><kbd>→</kbd> Bỏ Qua</div>
            <div><kbd>Esc</kbd> Xóa Chữ</div>
        </div>

        <div class="info-panel" id="info-panel">
            Đang tải dữ liệu từ server...
        </div>
    </div>

    <div class="toast" id="toast">✅ Đã lưu thành công!</div>

    <script>
        let metadata = [];
        let filteredIndices = [];
        let currentFilterIdx = 0;
        let currentFilter = 'unlabeled';
        let serverConfig = {};

        // Load config from server
        async function loadConfig() {
            try {
                const res = await fetch('/api/config');
                serverConfig = await res.json();
                document.getElementById('mode-badge').textContent = serverConfig.mode_name;
                document.getElementById('info-panel').innerHTML = 
                    `Thư mục làm việc: <b>${serverConfig.directory}</b><br>Tự động phát hiện cấu hình và tối ưu hóa hiển thị.`;
            } catch (e) {
                console.error("Failed to load config", e);
            }
        }

        // Load metadata/images from server
        async function loadMetadata() {
            const res = await fetch('/api/metadata');
            metadata = await res.json();
            applyFilter();
            updateStats();
        }

        function applyFilter() {
            filteredIndices = [];
            for (let i = 0; i < metadata.length; i++) {
                if (currentFilter === 'all') {
                    filteredIndices.push(i);
                } else if (currentFilter === 'unlabeled' && !metadata[i].text) {
                    filteredIndices.push(i);
                } else if (currentFilter === 'labeled' && metadata[i].text) {
                    filteredIndices.push(i);
                }
            }
            currentFilterIdx = 0;
            showCurrent();
        }

        function setFilter(filter) {
            currentFilter = filter;
            document.querySelectorAll('.filter-buttons button').forEach(b => b.classList.remove('active'));
            document.getElementById('btn-' + filter).classList.add('active');
            applyFilter();
        }

        function updateStats() {
            const total = metadata.length;
            const labeled = metadata.filter(m => m.text).length;
            document.getElementById('total-count').textContent = total;
            document.getElementById('labeled-count').textContent = labeled;
            document.getElementById('remaining-count').textContent = total - labeled;
            
            const percentage = total > 0 ? (labeled / total * 100) : 0;
            document.getElementById('progress-bar').style.width = percentage + '%';
        }

        function showCurrent() {
            const imgEl = document.getElementById('captcha-img');
            const filenameEl = document.getElementById('filename-label');
            const inputEl = document.getElementById('label-input');

            if (filteredIndices.length === 0) {
                filenameEl.textContent = 'Không còn ảnh nào khớp bộ lọc';
                imgEl.src = '';
                imgEl.alt = 'Đã hoàn thành gán nhãn!';
                inputEl.value = '';
                inputEl.disabled = true;
                return;
            }

            inputEl.disabled = false;
            const idx = filteredIndices[currentFilterIdx];
            const item = metadata[idx];
            filenameEl.textContent = `${item.filename} (${currentFilterIdx + 1}/${filteredIndices.length})`;
            
            // Append random query parameter to bypass cache and force reload (specifically useful for renaming mode)
            imgEl.src = '/images/' + item.filename + '?t=' + new Date().getTime();
            inputEl.value = item.text || '';
            inputEl.focus();
            inputEl.select();
        }

        async function saveLabel() {
            if (filteredIndices.length === 0) return;
            const idx = filteredIndices[currentFilterIdx];
            const item = metadata[idx];
            const input = document.getElementById('label-input');
            const text = input.value.trim().toUpperCase();

            if (text.length !== 5) {
                showToastError('❌ Nhãn phải chứa đúng 5 ký tự!');
                input.focus();
                return;
            }

            // Gửi lên server
            const res = await fetch('/api/save', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ index: item.index, filename: item.filename, text: text })
            });

            if (res.ok) {
                metadata[idx].text = text;
                showToast('✅ Đã lưu: ' + text);
                
                if (!serverConfig.csv_mode) {
                    // In renaming mode, renaming changes the filename, so we update the local item name
                    metadata[idx].filename = `map_${text}.png`;
                }

                updateStats();
                
                if (currentFilter === 'unlabeled') {
                    // Remove from active unlabeled list
                    filteredIndices.splice(currentFilterIdx, 1);
                    if (currentFilterIdx >= filteredIndices.length) {
                        currentFilterIdx = Math.max(0, filteredIndices.length - 1);
                    }
                    showCurrent();
                } else {
                    nextImage();
                }
            } else {
                showToastError('❌ Lỗi hệ thống: Không thể lưu nhãn');
            }
        }

        function nextImage() {
            if (currentFilterIdx < filteredIndices.length - 1) {
                currentFilterIdx++;
            }
            showCurrent();
        }

        function prevImage() {
            if (currentFilterIdx > 0) {
                currentFilterIdx--;
                showCurrent();
            }
        }

        function gotoImage() {
            const input = document.getElementById('goto-input');
            const val = parseInt(input.value);
            if (!isNaN(val) && val >= 1 && val <= filteredIndices.length) {
                currentFilterIdx = val - 1;
                showCurrent();
                input.value = '';
            } else {
                showToastError('❌ Số thứ tự không hợp lệ!');
            }
        }

        function showToast(msg) {
            const toast = document.getElementById('toast');
            toast.textContent = msg;
            toast.style.borderColor = 'var(--success)';
            toast.style.color = 'var(--success)';
            toast.style.background = 'rgba(0, 255, 136, 0.15)';
            toast.classList.add('show');
            setTimeout(() => toast.classList.remove('show'), 1500);
        }

        function showToastError(msg) {
            const toast = document.getElementById('toast');
            toast.textContent = msg;
            toast.style.borderColor = '#ff4d4d';
            toast.style.color = '#ff4d4d';
            toast.style.background = 'rgba(255, 77, 77, 0.15)';
            toast.classList.add('show');
            setTimeout(() => toast.classList.remove('show'), 2000);
        }

        // Keyboard shortcuts
        document.addEventListener('keydown', (e) => {
            if (e.key === 'Enter') {
                e.preventDefault();
                saveLabel();
            } else if (e.key === 'ArrowRight' && document.activeElement.id !== 'label-input') {
                nextImage();
            } else if (e.key === 'ArrowLeft' && document.activeElement.id !== 'label-input') {
                prevImage();
            } else if (e.key === 'Escape') {
                const input = document.getElementById('label-input');
                input.value = '';
                input.focus();
            }
        });

        // Initialize Web GUI
        async function init() {
            await loadConfig();
            await loadMetadata();
        }
        init();
    </script>
</body>
</html>
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Beautiful & Dual-Mode CAPTCHA Labeling Hub.")
    parser.add_argument(
        "--dir", "-d",
        type=str,
        help="Path to directory containing images/metadata.csv"
    )
    parser.add_argument(
        "--port", "-p",
        type=int,
        default=8080,
        help="Server port (default: 8080)"
    )
    args = parser.parse_args()

    # Determine data directory
    data_dir = None
    if args.dir:
        data_dir = Path(args.dir)
    else:
        # Smart fallback detection
        if DEFAULT_MAPS_DIR.exists():
            data_dir = DEFAULT_MAPS_DIR
        else:
            data_dir = DEFAULT_DATA_DIR

    if not data_dir.exists():
        print(f"[ERROR] Directory '{data_dir}' does not exist.")
        sys.exit(1)

    server = LabelingServer(("0.0.0.0", args.port), LabelHandler, data_dir)
    
    print(f"CAPTCHA Labeling Hub is Live!")
    print(f"   URL: http://localhost:{args.port}")
    print(f"   Network access: http://0.0.0.0:{args.port}")
    print(f"   Press Ctrl+C to stop the server.")
    print("-" * 50)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nServer stopped. Goodbye!")
        server.shutdown()


if __name__ == "__main__":
    main()

import express from 'express';
import multer from 'multer';
import { pipeline, env } from '@huggingface/transformers';
import path from 'path';
import fs from 'fs';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// Cấu hình môi trường cho local model
env.allowRemoteModels = false;
env.allowLocalModels = true;
env.localModelPath = path.join(__dirname, '../'); 

// --- CẤU HÌNH ĐẶC BIỆT CHO TERMUX/ANDROID ---
const wasmFolder = path.join(__dirname, 'node_modules', 'onnxruntime-web', 'dist');
env.backends.onnx.wasm.wasmPaths = `file://${wasmFolder}/`;
env.backends.onnx.wasm.proxy = false; 
env.backends.onnx.wasm.numThreads = 1;
// --------------------------------------------

const app = express();
const port = 5000;
const upload = multer({ storage: multer.memoryStorage() });
const tmpDir = path.join(__dirname, 'tmp');
if (!fs.existsSync(tmpDir)) fs.mkdirSync(tmpDir);

app.use(express.json({ limit: '10mb' }));
app.use(express.static(path.join(__dirname, 'public'))); 

// --- HỆ THỐNG LƯU LOG ---
const MAX_LOGS = 200;
const appLogs = [];
const originalLog = console.log;
const originalError = console.error;

function captureLog(type, args) {
    const msg = Array.from(args).map(a => typeof a === 'object' ? JSON.stringify(a) : a).join(' ');
    const logEntry = { time: new Date().toLocaleTimeString(), type, msg };
    appLogs.push(logEntry);
    if (appLogs.length > MAX_LOGS) appLogs.shift();
}

console.log = function(...args) { originalLog.apply(console, args); captureLog('info', args); };
console.error = function(...args) { originalError.apply(console, args); captureLog('error', args); };

// --- KHỞI TẠO MODEL ---
let captchaSolver;
let isReady = false;

async function initModel() {
    console.log('[SYSTEM] Đang khởi động AI Engine (Wasm Mode)...');
    try {
        captchaSolver = await pipeline('image-to-text', 'onnx_model', {
            device: 'cpu',
            dtype: 'fp32'
        });
        isReady = true;
        console.log('[SYSTEM] Tải mô hình TrOCR thành công! API đã sẵn sàng.');
    } catch (e) {
        console.error('[ERROR] Lỗi tải mô hình:', e.message);
    }
}

// --- REST API ENDPOINTS ---

// 1. Giải mã qua File Upload
app.post('/solve-file', upload.single('file'), async (req, res) => {
    if (!isReady) return res.status(503).json({ success: false, error: "Model đang tải" });
    if (!req.file) return res.status(400).json({ success: false, error: "Không tìm thấy file" });

    const tmpFilePath = path.join(tmpDir, `tmp_${Date.now()}_${req.file.originalname}`);
    try {
        const startTime = performance.now();
        console.log(`[API] Đang xử lý: ${req.file.originalname}`);
        
        // Lưu file tạm vào máy
        fs.writeFileSync(tmpFilePath, req.file.buffer);

        // Chạy suy luận với các tham số tối ưu cho CAPTCHA (5 ký tự)
        const result = await captchaSolver(tmpFilePath, {
            max_new_tokens: 10,
            do_sample: false,
            num_beams: 1, // Beam search = 1 để chạy nhanh nhất có thể
        });
        
        console.log('[DEBUG] Raw AI Output:', JSON.stringify(result));
        
        const rawText = result[0].generated_text || "";
        const text = rawText.replace(/[^a-zA-Z0-9]/g, '').toUpperCase();
        
        const timeTaken = (performance.now() - startTime).toFixed(2);
        console.log(`[API] => Kết quả: ${text} (Gốc: "${rawText}") - ${timeTaken}ms`);

        res.json({ success: true, captcha: text, inference_time_ms: parseFloat(timeTaken) });
    } catch (error) {
        console.error('[API ERROR]', error.message);
        res.status(500).json({ success: false, error: error.message });
    } finally {
        // Xóa file tạm ngay lập tức
        if (fs.existsSync(tmpFilePath)) fs.unlinkSync(tmpFilePath);
    }
});

// 2. Giải mã qua Base64
app.post('/solve-base64', async (req, res) => {
    if (!isReady) return res.status(503).json({ success: false, error: "Model đang tải" });
    if (!req.body.image_base64) return res.status(400).json({ success: false, error: "Thiếu base64" });

    const tmpFilePath = path.join(tmpDir, `tmp_b64_${Date.now()}.png`);
    try {
        const startTime = performance.now();
        console.log(`[API] Nhận request Base64`);
        
        let b64Data = req.body.image_base64;
        if (b64Data.includes(',')) b64Data = b64Data.split(',')[1];
        
        // Ghi file từ Base64
        fs.writeFileSync(tmpFilePath, Buffer.from(b64Data, 'base64'));

        const result = await captchaSolver(tmpFilePath);
        const text = result[0].generated_text.replace(/ /g, '').toUpperCase();
        
        const timeTaken = (performance.now() - startTime).toFixed(2);
        console.log(`[API] => Kết quả: ${text} (${timeTaken}ms)`);

        res.json({ success: true, captcha: text, inference_time_ms: parseFloat(timeTaken) });
    } catch (error) {
        console.error('[API ERROR]', error.message);
        res.status(500).json({ success: false, error: error.message });
    } finally {
        if (fs.existsSync(tmpFilePath)) fs.unlinkSync(tmpFilePath);
    }
});

app.get('/api/logs', (req, res) => res.json(appLogs));
app.get('/api/status', (req, res) => res.json({ status: isReady ? 'online' : 'starting', uptime: process.uptime(), memoryUsage: process.memoryUsage() }));
app.get('/health', (req, res) => res.json({ status: isReady ? 'healthy' : 'initializing', engine: 'transformers.js (Wasm)' }));

initModel().then(() => {
    app.listen(port, '0.0.0.0', () => console.log(`[SYSTEM] Server đang chạy tại cổng ${port}`));
});
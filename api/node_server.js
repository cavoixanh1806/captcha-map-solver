import express from 'express';
import multer from 'multer';
import { pipeline, env } from '@huggingface/transformers';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// Cấu hình môi trường cho local model
env.allowRemoteModels = false;
env.allowLocalModels = true;
env.localModelPath = path.join(__dirname, '../'); 

// --- CẤU HÌNH ĐẶC BIỆT CHO TERMUX/ANDROID (Fix ERR_UNSUPPORTED_ESM_URL_SCHEME) ---
// Ép dùng đường dẫn vật lý cục bộ cho Wasm engine
const wasmFolder = path.join(__dirname, 'node_modules', 'onnxruntime-web', 'dist');
env.backends.onnx.wasm.wasmPaths = `file://${wasmFolder}/`;
env.backends.onnx.wasm.proxy = false; 
env.backends.onnx.wasm.numThreads = 1; // Snapdragon 8s Gen 3 chạy cực nhanh ngay cả với 1 luồng, tránh lỗi Worker
// -------------------------------------------------------------------------------

const app = express();
const port = 5000;
const upload = multer({ storage: multer.memoryStorage() });

app.use(express.json({ limit: '10mb' }));
app.use(express.static(path.join(__dirname, 'public'))); 

// --- HỆ THỐNG LƯU LOG ĐỂ HIỂN THỊ LÊN WEBAPP ---
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

console.log = function(...args) {
    originalLog.apply(console, args);
    captureLog('info', args);
};
console.error = function(...args) {
    originalError.apply(console, args);
    captureLog('error', args);
};

// --- KHỞI TẠO MODEL ---
let captchaSolver;
let isReady = false;

async function initModel() {
    console.log('[SYSTEM] Đang khởi động AI Engine (Wasm Mode)...');
    try {
        captchaSolver = await pipeline('image-to-text', 'onnx_model', {
            device: 'cpu', // Trong Node.js, 'cpu' sẽ sử dụng Wasm backend
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
    if (!isReady) return res.status(503).json({ success: false, error: "Model đang tải, vui lòng thử lại sau" });
    if (!req.file) return res.status(400).json({ success: false, error: "Không tìm thấy file ảnh" });

    try {
        const startTime = performance.now();
        console.log(`[API] Đang xử lý file ảnh: ${req.file.originalname} (${req.file.size} bytes)`);
        
        // Truyền trực tiếp Buffer vào model (nhanh và tránh lỗi 404)
        const result = await captchaSolver(req.file.buffer);
        const text = result[0].generated_text.replace(/ /g, '').toUpperCase();
        
        const timeTaken = (performance.now() - startTime).toFixed(2);
        console.log(`[API] => Kết quả: ${text} (${timeTaken}ms)`);

        res.json({ success: true, captcha: text, inference_time_ms: parseFloat(timeTaken) });
    } catch (error) {
        console.error('[API ERROR]', error.message);
        res.status(500).json({ success: false, error: error.message });
    }
});

// 2. Giải mã qua chuỗi Base64
app.post('/solve-base64', async (req, res) => {
    if (!isReady) return res.status(503).json({ success: false, error: "Model đang tải" });
    if (!req.body.image_base64) return res.status(400).json({ success: false, error: "Thiếu trường image_base64" });

    try {
        const startTime = performance.now();
        console.log(`[API] Nhận request Base64 (độ dài: ${req.body.image_base64.length} chars)`);
        
        let b64Data = req.body.image_base64;
        if (b64Data.includes(',')) {
            b64Data = b64Data.split(',')[1];
        }
        
        // Chuyển Base64 thành Buffer trước khi đưa vào model
        const buffer = Buffer.from(b64Data, 'base64');

        const result = await captchaSolver(buffer);
        const text = result[0].generated_text.replace(/ /g, '').toUpperCase();
        
        const timeTaken = (performance.now() - startTime).toFixed(2);
        console.log(`[API] => Kết quả: ${text} (${timeTaken}ms)`);

        res.json({ success: true, captcha: text, inference_time_ms: parseFloat(timeTaken) });
    } catch (error) {
        console.error('[API ERROR]', error.message);
        res.status(500).json({ success: false, error: error.message });
    }
});

// 3. Lấy Logs cho WebApp
app.get('/api/logs', (req, res) => {
    res.json(appLogs);
});

// 4. Trạng thái Server
app.get('/api/status', (req, res) => {
    res.json({
        status: isReady ? 'online' : 'starting',
        uptime: process.uptime(),
        memoryUsage: process.memoryUsage()
    });
});

initModel().then(() => {
    app.listen(port, '0.0.0.0', () => {
        console.log(`[SYSTEM] TrOCR Local Server đang chạy tại cổng ${port}`);
    });
});
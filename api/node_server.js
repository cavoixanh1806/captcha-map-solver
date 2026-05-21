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
env.localModelPath = '../'; 

// --- CẤU HÌNH ĐẶC BIỆT CHO TERMUX/ANDROID ---
env.backends.onnx.wasm.proxy = false; 
env.backends.onnx.gpu = false;        
// --------------------------------------------

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
        
        const base64Data = req.file.buffer.toString('base64');
        const dataUrl = `data:${req.file.mimetype};base64,${base64Data}`;

        const result = await captchaSolver(dataUrl);
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
        
        const base64Str = req.body.image_base64.startsWith('data:') 
            ? req.body.image_base64 
            : `data:image/png;base64,${req.body.image_base64}`;

        const result = await captchaSolver(base64Str);
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
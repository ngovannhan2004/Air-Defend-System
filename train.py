import cv2
import time
import json
import serial
import threading
import numpy as np
from flask import Flask, Response, render_template_string
from ultralytics import YOLO

app = Flask(__name__)

# ==============================================================================
# CLASS 1: QUẢN LÝ KẾT NỐI SERIAL PHẦN CỨNG (GIỮ NGUYÊN LOGIC CŨ)
# ==============================================================================
class SerialManager:
    def __init__(self, port="COM5", baudrate=115200):
        self.distance = 0
        self.motion = 0
        self.ser = serial.Serial(
            port,
            baudrate,
            timeout=0.1
        )
        self.running = True
        self.thread = threading.Thread(
            target=self.read_loop,
            daemon=True
        )
        self.thread.start()

    def read_loop(self):
        while self.running:
            try:
                line = self.ser.readline().decode().strip()
                if line.startswith("D:"):
                    parts = line.split(",")
                    self.distance = int(parts[0].split(":")[1])
                    self.motion = int(parts[1].split(":")[1])
            except:
                pass

    def send_servo(self, pan, tilt):
        cmd = f"P:{pan},T:{tilt}\n"
        try:
            self.ser.write(cmd.encode())
        except:
            pass

    def close(self):
        self.running = False
        if self.ser.is_open:
            self.ser.close()

# ==============================================================================
# CLASS 2: THUẬT TOÁN ĐIỀU KHIỂN PAN-TILT TRACKER (GIỮ NGUYÊN LOGIC CŨ)
# ==============================================================================
class PanTiltTracker:
    def __init__(self):
        self.pan = 90
        self.tilt = 90
        self.last_cx = 320
        self.last_cy = 320
        self.KP = 0.1
        self.MAX_STEP = 3
        self.HFOV = 62
        self.VFOV = 48

    def update(self, cx, cy):
        cx = int(0.7 * self.last_cx + 0.3 * cx)
        cy = int(0.7 * self.last_cy + 0.3 * cy)
        self.last_cx = cx
        self.last_cy = cy

        error_x = cx - 320
        error_y = cy - 320

        if abs(error_x) < 10: error_x = 0
        if abs(error_y) < 10: error_y = 0

        angle_x = error_x * (self.HFOV / 640)
        angle_y = error_y * (self.VFOV / 640)

        delta_pan = angle_x * self.KP
        delta_tilt = angle_y * self.KP

        delta_pan = np.clip(delta_pan, -self.MAX_STEP, self.MAX_STEP)
        delta_tilt = np.clip(delta_tilt, -self.MAX_STEP, self.MAX_STEP)

        self.pan += delta_pan
        self.tilt += delta_tilt

        self.pan = np.clip(self.pan, 0, 180)
        self.tilt = np.clip(self.tilt, 0, 180)

        return int(self.pan), int(self.tilt)

# ==============================================================================
# CẤU HÌNH HỆ THỐNG & KHỞI TẠO ĐỐI TƯỢNG
# ==============================================================================
PORT = "COM5"
model = YOLO("best.pt")

# Khởi tạo kết nối phần cứng
try:
    serial_mgr = SerialManager(port=PORT, baudrate=115200)
except Exception as e:
    print(f"⚠️ Không thể mở cổng {PORT} ({e}). Chạy chế độ giả lập dữ liệu.")
    serial_mgr = None

tracker = PanTiltTracker()

# Mở camera (Mặc định dùng ID 1 như code cũ của bạn)
cap = cv2.VideoCapture(1)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 640)

class_names = {
    0: "Helicopter",
    1: "Jet",
    2: "Rocket"
}

# Biến lưu trữ trạng thái đồng bộ dữ liệu thời gian thực lên giao diện Web
system_state = {
    "distance": 0,
    "motion": 0,
    "status": "IDLE",
    "pan": 90,
    "tilt": 90,
    "label": "-",
    "conf": 0.0,
    "log_msg": ""
}

# ==============================================================================
# LUỒNG XỬ LÝ CORE LOGIC (CAMERA + AI + ĐIỀU KHIỂN PHẦN CỨNG)
# ==============================================================================
def generate_frames():
    global system_state
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (640, 640))

        # Đọc dữ liệu từ phần cứng thật
        if serial_mgr:
            distance = serial_mgr.distance
            motion = serial_mgr.motion
        else:
            distance = 1250  # Giá trị mô phỏng nếu chưa cắm mạch
            motion = 1

        status = "IDLE"
        label = "-"
        conf = 0.0
        pan = tracker.pan
        tilt = tracker.tilt
        target_found = False

        # Nhận diện vật thể bằng YOLO
        results = model(frame, conf=0.5, verbose=False)

        if motion == 1:
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    conf = float(box.conf)
                    cls = int(box.cls)
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    
                    cx = (x1 + x2) // 2
                    cy = (y1 + y2) // 2

                    label = class_names.get(cls, "Unknown")

                    if distance < 5000:
                        target_found = True
                        status = "TRACKING"

                        # Tính toán góc quay và gửi lệnh xuống mạch thật
                        pan, tilt = tracker.update(cx, cy)
                        if serial_mgr:
                            serial_mgr.send_servo(pan, tilt)

                        # Vẽ khung nhận diện lên camera (Đẩy trực tiếp lên giao diện Web)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(
                            frame, f"{label} {conf:.2f}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
                        )
                        cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)
                        break

        if not target_found:
            status = "SEARCHING" if motion == 1 else "IDLE"

        # Vẽ tâm ngắm Crosshair cố định lên màn hình camera
        cv2.line(frame, (320, 0), (320, 640), (255, 255, 255), 1)
        cv2.line(frame, (0, 320), (640, 320), (255, 255, 255), 1)

        # Tạo chuỗi thông báo log real-time dựa theo dữ liệu thực tế
        current_time = time.strftime("%H:%M:%S", time.localtime())
        log_msg = ""
        if status == "TRACKING":
            log_msg = f"[{current_time}] [CẢNH BÁO] Phát hiện {label}! Khoảng cách: {distance}mm. Servo điều hướng: X={pan}°, Y={tilt}°"
        elif status == "SEARCHING":
            log_msg = f"[{current_time}] [RADAR] Đang quét không gian tọa độ... Trạng thái: Tìm kiếm mục tiêu."

        # Cập nhật toàn bộ thông số thực tế vào State tổng
        system_state = {
            "distance": distance,
            "motion": motion,
            "status": status,
            "pan": int(pan),
            "tilt": int(tilt),
            "label": label,
            "conf": round(conf, 2),
            "log_msg": log_msg
        }

        # Nén Frame ảnh thành định dạng .jpg để truyền phát qua môi trường mạng Web
        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            continue
            
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

# ==============================================================================
# ĐƯỜNG DẪN MẠNG API ĐỒNG BỘ WEB (FLASK SERVER ENDPOINTS)
# ==============================================================================
@app.route('/')
def index():
    return render_template_string(HTML_LAYOUT)

@app.route('/video_feed')
def video_feed():
    return Response(
        generate_frames(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )

@app.route('/telemetry')
def telemetry():
    def event_stream():
        while True:
            # Liên tục gửi gói tin JSON chứa cảm biến thật xuống giao diện sau mỗi 100ms
            json_data = json.dumps(system_state)
            yield f"data: {json_data}\n\n"
            time.sleep(0.1)
    return Response(event_stream(), mimetype='text/event-stream')

# ==============================================================================
# GIAO DIỆN ĐỒ HỌA CYBERPUNK (HTML/CSS/JS) ĐỒNG BỘ DỮ LIỆU THẬT
# ==============================================================================
HTML_LAYOUT = """
<!DOCTYPE html>
<html lang="vi">
<head>
    <meta charset="UTF-8">
    <title>SMART TRACKING & MONITORING SYSTEM</title>
    <style>
        * { box-sizing: border-box; }
        body {
            background-color: #030a10;
            color: #00f0ff;
            font-family: 'Courier New', Courier, monospace;
            margin: 0; padding: 15px;
            overflow-x: hidden;
        }
        
        .header-panel {
            border: 1px solid #005f73;
            background: rgba(2, 17, 27, 0.8);
            padding: 10px 20px;
            display: flex;
            justify-content: space-between; align-items: center;
            margin-bottom: 15px;
            box-shadow: 0 0 15px rgba(0, 240, 255, 0.1);
        }
        .header-title h1 { margin: 0; font-size: 22px; letter-spacing: 2px; text-shadow: 0 0 8px #00f0ff; }
        .header-title small { color: #7094b0; font-size: 11px; }
        .time-display { font-size: 18px; font-weight: bold; color: #ffffff; }
        .status-dot { color: #00ff66; font-size: 12px; }

        .main-dashboard {
            display: grid;
            grid-template-columns: 1.3fr 1fr;
            gap: 15px;
        }

        .panel-box {
            border: 1px solid #005f73;
            background: rgba(2, 17, 27, 0.8);
            padding: 12px; position: relative;
        }
        .panel-header {
            font-size: 12px; color: #7094b0; border-bottom: 1px solid #003f5c;
            padding-bottom: 6px; margin-bottom: 12px; text-transform: uppercase;
        }

        .camera-container { width: 100%; background: #000; border: 1px solid #002b3d; }
        .camera-container img { width: 100%; height: auto; max-height: 520px; display: block; object-fit: contain; }

        .right-column { display: flex; flex-direction: column; gap: 15px; }
        .radar-scanner-box { flex: 1; display: flex; flex-direction: column; align-items: center; justify-content: center; padding: 20px 0; }
        
        /* Hiệu ứng đồ họa Radar vòng xoay */
        .radar-screen {
            width: 240px; height: 240px; border: 2px solid #005f73; border-radius: 50%;
            position: relative; background: radial-gradient(circle, transparent 35%, rgba(0, 95, 115, 0.15) 100%);
            margin-bottom: 10px; overflow: hidden;
        }
        .radar-circle-line { position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%); border: 1px dashed rgba(0, 240, 255, 0.2); border-radius: 50%; }
        .radar-circle-1 { width: 160px; height: 160px; }
        .radar-circle-2 { width: 80px; height: 80px; }
        .radar-cross-h { position: absolute; top: 50%; left: 5%; right: 5%; height: 1px; background: rgba(0, 240, 255, 0.15); }
        .radar-cross-v { position: absolute; left: 50%; top: 5%; bottom: 5%; width: 1px; background: rgba(0, 240, 255, 0.15); }
        .radar-sweep-line {
            position: absolute; width: 50%; height: 50%; bottom: 50%; right: 50%;
            background: linear-gradient(45deg, rgba(0, 240, 255, 0.35) 0%, transparent 70%);
            transform-origin: bottom right; animation: radar-rotate 3.5s linear infinite;
            border-right: 1px solid rgba(0, 240, 255, 0.5);
        }
        @keyframes radar-rotate { from { transform: rotate(0deg); } to { transform: rotate(360deg); } }
        
        /* Các mục tiêu nhấp nháy mô phỏng trên Radar nền */
        .radar-blip-1 { position: absolute; top: 35%; left: 65%; width: 6px; height: 6px; background: #ffaa00; border-radius: 50%; box-shadow: 0 0 8px #ffaa00; }
        .radar-blip-2 { position: absolute; top: 60%; left: 75%; width: 6px; height: 6px; background: #ffaa00; border-radius: 50%; box-shadow: 0 0 8px #ffaa00; }

        /* Khung hiển thị dữ liệu thẻ thông số */
        .telemetry-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; }
        .data-card { background: rgba(4, 28, 44, 0.6); border-left: 3px solid #00f0ff; padding: 8px 12px; }
        .card-title { font-size: 10px; color: #7094b0; margin-bottom: 4px; }
        .card-value { font-size: 20px; font-weight: bold; color: #ffffff; }
        .card-value small { font-size: 11px; color: #00f0ff; margin-left: 2px; }
        
        .status-alert { color: #ff3333 !important; text-shadow: 0 0 8px rgba(255, 51, 51, 0.6); animation: blinker 1.5s linear infinite; }
        @keyframes blinker { 50% { opacity: 0.4; } }

        .log-panel { grid-column: span 2; background: #010508; border: 1px solid #005f73; height: 110px; padding: 8px 12px; font-size: 12px; color: #92b0c5; overflow-y: auto; }
        .log-line { margin-bottom: 3px; border-bottom: 1px solid rgba(0, 95, 115, 0.1); padding-bottom: 2px; }
    </style>
</head>
<body>

    <div class="header-panel">
        <div class="header-title">
            <h1>SMART TRACKING & MONITORING SYSTEM</h1>
            <small>HOST: RASPBERRY PI 4 | NODE: ESP32</small>
        </div>
        <div class="header-status">
            <div class="time-display" id="live-clock">00:00:00</div>
            <div class="status-dot">● SYSTEM ONLINE</div>
        </div>
    </div>

    <div class="main-dashboard">
        <div class="panel-box">
            <div class="panel-header">● LIVE STREAM [CAM_01 | 640x640 REAL-TIME]</div>
            <div class="camera-container">
                <img src="/video_feed" alt="Đang kết nối luồng camera phát trực tuyến...">
            </div>
        </div>

        <div class="right-column">
            <div class="panel-box radar-scanner-box">
                <div class="panel-header" style="width:100%; position:absolute; top:12px; left:12px;">RADAR MICRO-WAVE SCANNER (RCWL-0516)</div>
                <div class="radar-screen">
                    <div class="radar-sweep-line"></div>
                    <div class="radar-circle-line radar-circle-1"></div>
                    <div class="radar-circle-line radar-circle-2"></div>
                    <div class="radar-cross-h"></div>
                    <div class="radar-cross-v"></div>
                    <div class="radar-blip-1"></div>
                    <div class="radar-blip-2"></div>
                </div>
            </div>

            <div class="panel-box">
                <div class="telemetry-grid">
                    <div class="data-card">
                        <div class="card-title">KHOẢNG CÁCH (VL53L0X)</div>
                        <div class="card-value" id="val-distance">0 <small>mm</small></div>
                    </div>
                    <div class="data-card">
                        <div class="card-title">TRẠNG THÁI HỆ THỐNG</div>
                        <div class="card-value" id="val-status">IDLE</div>
                    </div>
                    <div class="data-card">
                        <div class="card-title">SERVO NGANG (PAN)</div>
                        <div class="card-value" id="val-pan">0°</div>
                    </div>
                    <div class="data-card">
                        <div class="card-title">SERVO DỌC (TILT)</div>
                        <div class="card-value" id="val-tilt">0°</div>
                    </div>
                </div>
            </div>
        </div>

        <div class="log-panel" id="terminal-log">
            <div class="log-line" style="color: #00ff66;">[SYSTEM] Đồng bộ hóa toàn bộ luồng dữ liệu và thiết bị phần cứng thành công...</div>
        </div>
    </div>

    <script>
        function updateClock() {
            document.getElementById('live-clock').innerText = new Date().toLocaleTimeString();
        }
        setInterval(updateClock, 1000); updateClock();

        // KẾT NỐI VÀ LẮNG NGHE DỮ LIỆU CẢM BIẾN THẬT TỪ BACKEND PYTHON ĐẨY LÊN
        const telemetrySource = new EventSource("/telemetry");
        const logPanel = document.getElementById("terminal-log");
        let lastLog = "";

        telemetrySource.onmessage = function(event) {
            const state = JSON.parse(event.data);

            // Gán giá trị khoảng cách và góc servo thực tế từ mạch phần cứng lên giao diện Web
            document.getElementById("val-distance").innerHTML = `${state.distance} <small>mm</small>`;
            document.getElementById("val-pan").innerText = `${state.pan}°`;
            document.getElementById("val-tilt").innerText = `${state.tilt}°`;

            // Thay đổi màu sắc và chữ trạng thái động trên giao diện Web theo kết quả tính toán AI
            const statusCard = document.getElementById("val-status");
            if (state.status === "TRACKING") {
                statusCard.innerText = `BÁM ĐUỔI: ${state.label.toUpperCase()}`;
                statusCard.className = "card-value status-alert";
            } else if (state.status === "SEARCHING") {
                statusCard.innerText = "ĐANG QUÉT MỤC TIÊU";
                statusCard.className = "card-value";
                statusCard.style.color = "#ffaa00";
            } else {
                statusCard.innerText = "SẴN SÀNG (IDLE)";
                statusCard.className = "card-value";
                statusCard.style.color = "#ffffff";
            }

            // Đẩy văn bản nhật ký hệ thống real-time xuống khung log dưới đáy màn hình
            if (state.log_msg && state.log_msg !== lastLog) {
                lastLog = state.log_msg;
                const newLogLine = document.createElement("div");
                newLogLine.className = "log-line";
                newLogLine.innerText = state.log_msg;
                
                if (state.status === "TRACKING") {
                    newLogLine.style.color = "#ff3333";
                }
                
                logPanel.appendChild(newLogLine);
                logPanel.scrollTop = logPanel.scrollHeight; // Tự cuộn khung log xuống dưới cùng
            }
        };
    </script>
</body>
</html>
"""

if __name__ == '__main__':
    # Khởi chạy cục bộ hệ thống Web Server tại địa chỉ cổng mạng 5500 giống hệt hình ảnh của bạn
    app.run(host='0.0.0.0', port=5500, debug=False)
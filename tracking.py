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
# CLASS 1: KẾT NỐI VÀ ĐỌC DỮ LIỆU CẢM BIẾN / SERVO QUA SERIAL
# ==============================================================================
class SerialManager:
    def __init__(self, port="COM5", baudrate=115200):
        self.distance = 0
        self.motion = 0
        try:
            self.ser = serial.Serial(port, baudrate, timeout=0.1)
            self.running = True
            self.thread = threading.Thread(target=self.read_loop, daemon=True)
            self.thread.start()
            print(f"✅ Thiết bị phần cứng đã kết nối tại cổng {port}")
        except Exception as e:
            print(f"⚠️ Chế độ GIẢ LẬP: Không tìm thấy cổng {port} ({e})")
            self.ser = None
            self.running = False

    def read_loop(self):
        while self.running and self.ser:
            try:
                if self.ser.in_waiting > 0:
                    line = self.ser.readline().decode().strip()
                    if line.startswith("D:"):
                        parts = line.split(",")
                        self.distance = int(parts[0].split(":")[1])
                        self.motion = int(parts[1].split(":")[1])
            except:
                pass

    def send_servo(self, pan, tilt):
        if self.ser and self.ser.is_open:
            cmd = f"P:{int(pan)},T:{int(tilt)}\n"
            try:
                self.ser.write(cmd.encode())
            except:
                pass

    def close(self):
        self.running = False
        if self.ser and self.ser.is_open:
            self.ser.close()

# ==============================================================================
# CLASS 2: BỘ ĐIỀU KHIỂN PAN-TILT TRACKER
# ==============================================================================
class PanTiltTracker:
    def __init__(self):
        self.pan = 90
        self.tilt = 90
        self.last_cx = 480
        self.last_cy = 270
        self.KP = 0.25
        self.MAX_STEP = 3
        self.HFOV = 62
        self.VFOV = 48

    def update(self, cx, cy, frame_w=960, frame_h=540):
        center_x = frame_w // 2
        center_y = frame_h // 2

        cx = int(0.7 * self.last_cx + 0.3 * cx)
        cy = int(0.7 * self.last_cy + 0.3 * cy)
        self.last_cx = cx
        self.last_cy = cy

        error_x = cx - center_x
        error_y = cy - center_y

        if abs(error_x) < 10: error_x = 0
        if abs(error_y) < 10: error_y = 0

        angle_x = error_x * (self.HFOV / frame_w)
        angle_y = error_y * (self.VFOV / frame_h)

        delta_pan = angle_x * self.KP
        delta_tilt = angle_y * self.KP

        delta_pan = np.clip(delta_pan, -self.MAX_STEP, self.MAX_STEP)
        delta_tilt = np.clip(delta_tilt, -self.MAX_STEP, self.MAX_STEP)

        # Đảo ngược hướng dịch chuyển nếu servo quay ngược hướng camera
        self.pan -= delta_pan 
        self.tilt += delta_tilt

        self.pan = np.clip(self.pan, 0, 180)
        self.tilt = np.clip(self.tilt, 0, 180)

        return int(self.pan), int(self.tilt)

# ==============================================================================
# KHỞI TẠO CẤU HÌNH HỆ THỐNG
# ==============================================================================
PORT = "COM5"            
model = YOLO("best.pt") 
tracker = PanTiltTracker()
serial_mgr = SerialManager(port=PORT, baudrate=115200)

cap = cv2.VideoCapture(0) 
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

class_names = {0: "Helicopter", 1: "Jet", 2: "Rocket"}

system_state = {
    "distance": 0,
    "motion": 0,
    "status": "INITIALIZING",
    "pan": 90,
    "tilt": 90,
    "label": "-",
    "conf": 0.0,
    "log_msg": "[SYSTEM] Đang khởi động luồng hệ thống..."
}

# ==============================================================================
# LUỒNG XỬ LÝ AI VÀ TRUYỀN DỮ LIỆU TELEMETRY
# ==============================================================================
def generate_frames():
    global system_state
    
    while True:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.03)
            continue

        TARGET_W = 960
        TARGET_H = 540
        frame = cv2.resize(frame, (TARGET_W, TARGET_H))

        if serial_mgr and serial_mgr.ser:
            distance = serial_mgr.distance
            motion = serial_mgr.motion
        else:
            distance = int(280 + 30 * np.sin(time.time()))
            motion = 1

        status = "IDLE"
        label = "-"
        conf = 0.0
        pan = tracker.pan
        tilt = tracker.tilt
        target_found = False

        results = model(frame, conf=0.5, imgsz=640, verbose=False)

        for result in results:
            boxes = result.boxes
            for box in boxes:
                conf = float(box.conf)
                cls = int(box.cls)
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2
                label = class_names.get(cls, "Target")

                target_found = True
                status = "TRACKING"

                pan, tilt = tracker.update(cx, cy, TARGET_W, TARGET_H)
                serial_mgr.send_servo(pan, tilt)

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    frame, f"{label} {conf:.2f}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2
                )
                cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)
                break
            if target_found: break

        if not target_found:
            if motion == 1 or (0 < distance < 600):
                status = "SEARCHING"
            else:
                status = "IDLE"

        # Vẽ lưới chữ thập ngắm tâm camera
        mid_x = TARGET_W // 2
        mid_y = TARGET_H // 2
        cv2.line(frame, (mid_x, 0), (mid_x, TARGET_H), (255, 255, 255), 1)
        cv2.line(frame, (0, mid_y), (TARGET_W, mid_y), (255, 255, 255), 1)

        current_time = time.strftime("%H:%M:%S", time.localtime())
        if status == "TRACKING":
            log_msg = f"[{current_time}] [CẢNH BÁO] Phát hiện {label}! Khoảng cách: {distance}mm. Góc: X={int(pan)}°"
        else:
            log_msg = f"[{current_time}] [SYSTEM] Trạng thái Quét tìm kiếm tự động..."

        system_state = {
            "distance": int(distance),
            "motion": int(motion),
            "status": status,
            "pan": float(pan),
            "tilt": float(tilt),
            "label": label,
            "conf": round(conf, 2),
            "log_msg": log_msg
        }

        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret: continue
            
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

@app.route('/')
def index():
    return render_template_string(HTML_LAYOUT)

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/telemetry')
def telemetry():
    def event_stream():
        while True:
            json_data = json.dumps(system_state)
            yield f"data: {json_data}\n\n"
            time.sleep(0.05)
    return Response(event_stream(), mimetype='text/event-stream')

# ==============================================================================
# GIAO DIỆN ĐỒ HỌA FRONT-END CAO CẤP (ĐÃ FIX TRIỆT ĐỂ TOÁN TỌA ĐỘ)
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
            overflow: hidden;
        }
        
        .header-panel {
            border: 1px solid #005f73;
            background: rgba(2, 17, 27, 0.8);
            padding: 10px 20px;
            display: flex;
            justify-content: space-between; align-items: center;
            margin-bottom: 15px;
        }
        .header-title h1 { margin: 0; font-size: 22px; letter-spacing: 2px; text-shadow: 0 0 8px #00f0ff; }
        .time-display { font-size: 18px; font-weight: bold; color: #ffffff; }

        .main-dashboard {
            display: grid;
            grid-template-columns: 1.3fr 0.9fr;
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

        .camera-container { 
            width: 100%; background: #000000; border: 1px solid #002b3d;
            display: flex; justify-content: center; align-items: center;
            aspect-ratio: 16 / 9; overflow: hidden;
        }
        .camera-container img { width: 100%; height: 100%; object-fit: contain; }

        .right-column { display: flex; flex-direction: column; gap: 15px; }
        .radar-scanner-box { 
            flex: 1; display: flex; flex-direction: column; 
            align-items: center; justify-content: center; padding: 40px 0 20px 0; 
        }
        
        #radarCanvas { background-color: transparent; display: block; margin: 0 auto; }

        .telemetry-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; }
        .data-card { background: rgba(4, 28, 44, 0.6); border-left: 3px solid #00f0ff; padding: 8px 12px; }
        .card-title { font-size: 10px; color: #7094b0; margin-bottom: 4px; }
        .card-value { font-size: 18px; font-weight: bold; color: #ffffff; }
        .card-value small { font-size: 11px; color: #00f0ff; }
        
        .status-alert { color: #ff3333 !important; text-shadow: 0 0 8px rgba(255, 51, 51, 0.6); }

        .log-panel { grid-column: span 2; background: #010508; border: 1px solid #005f73; height: 110px; padding: 8px 12px; font-size: 12px; color: #92b0c5; overflow-y: auto; }
        .log-line { margin-bottom: 3px; font-size: 11px; }
    </style>
</head>
<body>

    <div class="header-panel">
        <div class="header-title">
            <h1>SMART TRACKING & MONITORING SYSTEM</h1>
        </div>
        <div class="time-display" id="live-clock">00:00:00</div>
    </div>

    <div class="main-dashboard">
        <div class="panel-box">
            <div class="panel-header">● LIVE STREAM DETECTOR</div>
            <div class="camera-container">
                <img src="/video_feed">
            </div>
        </div>

        <div class="right-column">
            <div class="panel-box radar-scanner-box">
                <div class="panel-header" style="width:100%; position:absolute; top:12px; left:12px;">RADAR REALTIME TARGET POSITION</div>
                <canvas id="radarCanvas" width="280" height="280"></canvas>
            </div>

            <div class="panel-box">
                <div class="telemetry-grid">
                    <div class="data-card">
                        <div class="card-title">KHOẢNG CÁCH (VL53L0X)</div>
                        <div class="card-value" id="val-distance">0 <small>mm</small></div>
                    </div>
                    <div class="data-card">
                        <div class="card-title">TRẠNG THÁI RADAR</div>
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

        <div class="log-panel" id="terminal-log"></div>
    </div>

    <script>
        function updateClock() {
            document.getElementById('live-clock').innerText = new Date().toLocaleTimeString();
        }
        setInterval(updateClock, 1000);

        const canvas = document.getElementById("radarCanvas");
        const ctx = canvas.getContext("2d");
        const centerX = canvas.width / 2;
        const centerY = canvas.height / 2;
        const maxRadarRadius = canvas.width / 2 - 15; 

        // 3 VÒNG TRÒN MỐC KHOẢNG CÁCH
        const R_100mm = 100; 
        const R_300mm = 300; 
        const R_600mm = 600; 

        let currentPanAngle = 90;
        let currentDistance = 0;
        let systemStatus = "IDLE";
        let sweepAngle = 0;

        function drawRadar() {
            ctx.clearRect(0, 0, canvas.width, canvas.height);

            const radius600 = maxRadarRadius;
            const radius300 = maxRadarRadius * (R_300mm / R_600mm);
            const radius100 = maxRadarRadius * (R_100mm / R_600mm);

            ctx.strokeStyle = "rgba(0, 240, 255, 0.18)";
            ctx.lineWidth = 1;

            // 1. Vẽ vòng tròn trong (100mm)
            ctx.beginPath(); ctx.arc(centerX, centerY, radius100, 0, 2 * Math.PI);
            ctx.setLineDash([3, 3]); ctx.stroke();

            // 2. Vẽ vòng tròn giữa (300mm)
            ctx.beginPath(); ctx.arc(centerX, centerY, radius300, 0, 2 * Math.PI);
            ctx.setLineDash([]); ctx.stroke();

            // 3. Vẽ vòng tròn ngoài (600mm)
            ctx.strokeStyle = "rgba(0, 240, 255, 0.4)";
            ctx.beginPath(); ctx.arc(centerX, centerY, radius600, 0, 2 * Math.PI);
            ctx.stroke();

            // Trục ngắm chữ thập
            ctx.strokeStyle = "rgba(0, 240, 255, 0.15)";
            ctx.beginPath(); ctx.moveTo(centerX - radius600, centerY); ctx.lineTo(centerX + radius600, centerY); ctx.stroke();
            ctx.beginPath(); ctx.moveTo(centerX, centerY - radius600); ctx.lineTo(centerX, centerY + radius600); ctx.stroke();

            // Ghi text mốc khoảng cách
            ctx.fillStyle = "rgba(0, 240, 255, 0.6)";
            ctx.font = "9px Courier New";
            ctx.fillText("100mm", centerX + 5, centerY - radius100 + 10);
            ctx.fillText("300mm", centerX + 5, centerY - radius300 + 10);
            ctx.fillText("600mm", centerX + 5, centerY - radius600 + 10);

            // =================================================================
            // SỬA GÓC LƯỢNG GIÁC: Khớp tuyệt đối góc xoay của Servo vật lý
            // =================================================================
            let targetRad = (currentPanAngle * Math.PI) / 180;

            // Vẽ tia quét dải quạt tự động
            if (systemStatus === "TRACKING") {
                ctx.strokeStyle = "rgba(0, 255, 102, 0.7)";
                ctx.lineWidth = 2;
                ctx.beginPath();
                ctx.moveTo(centerX, centerY);
                ctx.lineTo(centerX + radius600 * Math.cos(targetRad), centerY - radius600 * Math.sin(targetRad));
                ctx.stroke();
            } else {
                sweepAngle += 0.04;
                ctx.save();
                ctx.translate(centerX, centerY);
                ctx.rotate(-targetRad + Math.sin(sweepAngle) * 0.2); 
                let gradient = ctx.createRadialGradient(0, 0, 0, 0, 0, radius600);
                gradient.addColorStop(0, "rgba(0, 240, 255, 0)");
                gradient.addColorStop(1, "rgba(0, 240, 255, 0.15)");
                ctx.fillStyle = gradient;
                ctx.beginPath(); ctx.moveTo(0, 0); ctx.arc(0, 0, radius600, -0.2, 0.2); ctx.closePath(); ctx.fill();
                ctx.restore();
            }

            // =================================================================
            // ÉP HIỂN THỊ CHẤM ĐỎ TRONG PHẠM VI 3 VÒNG TRÒN
            // =================================================================
            // Giới hạn khoảng cách hiển thị thực tế từ cảm biến (Ví dụ: 274mm)
            let checkDist = currentDistance;
            if (checkDist > 0 && checkDist <= 800) { 
                // Nếu khoảng cách vượt mốc 600mm biên, khóa điểm lại ở rìa ngoài cùng để không bị ẩn mất
                if (checkDist > R_600mm) checkDist = R_600mm;

                // Tính toán bán kính Pixel từ tỉ lệ khoảng cách mm
                let mappedRadius = maxRadarRadius * (checkDist / R_600mm);
                
                // Thuật toán tọa độ Descartes hướng góc phẳng chuẩn
                let pointX = centerX + mappedRadius * Math.cos(targetRad);
                let pointY = centerY - mappedRadius * Math.sin(targetRad);

                // Tiến hành kết xuất chấm đỏ rực rỡ lên màn hình canvas
                ctx.save();
                ctx.shadowBlur = 15;
                ctx.shadowColor = "#ff0000";
                ctx.fillStyle = "#ff0000";
                ctx.beginPath();
                ctx.arc(pointX, pointY, 7, 0, 2 * Math.PI); 
                ctx.fill();
                ctx.restore();
            }
        }

        function tick() {
            drawRadar();
            requestAnimationFrame(tick);
        }
        tick();

        // ĐỒNG BỘ TELEMETRY DỮ LIỆU
        const telemetrySource = new EventSource("/telemetry");
        const logPanel = document.getElementById("terminal-log");
        let lastLog = "";

        telemetrySource.onmessage = function(event) {
            const state = JSON.parse(event.data);

            currentDistance = parseInt(state.distance);
            currentPanAngle = parseFloat(state.pan);
            systemStatus = state.status;

            document.getElementById("val-distance").innerHTML = `${state.distance} <small>mm</small>`;
            document.getElementById("val-pan").innerText = `${Math.round(state.pan)}°`;
            document.getElementById("val-tilt").innerText = `${Math.round(state.tilt)}°`;

            const statusCard = document.getElementById("val-status");
            if (state.status === "TRACKING") {
                statusCard.innerText = `BÁM: ${state.label.toUpperCase()}`;
                statusCard.className = "card-value status-alert";
            } else {
                statusCard.innerText = "ĐANG TÌM KIẾM";
                statusCard.className = "card-value";
                statusCard.style.color = "#ffaa00";
            }

            if (state.log_msg && state.log_msg !== lastLog) {
                lastLog = state.log_msg;
                const newLogLine = document.createElement("div");
                newLogLine.className = "log-line";
                newLogLine.innerText = state.log_msg;
                if (state.status === "TRACKING") newLogLine.style.color = "#ff3333";
                logPanel.appendChild(newLogLine);
                logPanel.scrollTop = logPanel.scrollHeight;
            }
        };
    </script>
</body>
</html>
"""

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5500, debug=False)
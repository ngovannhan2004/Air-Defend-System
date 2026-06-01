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
# CLASS 1: QUẢN LÝ KẾT NỐI SERIAL PHẦN CỨNG
# ==============================================================================
class SerialManager:
    def __init__(self, port="COM5", baudrate=115200):
        self.distance = 8190  # Mặc định ban đầu ở trạng thái xa vô tận
        self.motion = 0
        try:
            self.ser = serial.Serial(port, baudrate, timeout=0.1)
            self.running = True
            self.thread = threading.Thread(target=self.read_loop, daemon=True)
            self.thread.start()
            print(f"✅ Đã kết nối phần cứng tại cổng: {port}")
        except Exception as e:
            print(f"⚠️ Chế độ giả lập tự động do không thấy cổng {port}")
            self.ser = None
            self.running = False

    def read_loop(self):
        while self.running and self.ser:
            try:
                if self.ser.in_waiting > 0:
                    line = self.ser.readline().decode().strip()
                    if line.startswith("D:"):
                        # Chuỗi nhận từ ESP32 có dạng: "D:khoang_cach,M:chuyen_dong"
                        parts = line.split(",")
                        self.distance = int(parts[0].split(":")[1])
                        self.motion = int(parts[1].split(":")[1])
            except:
                pass

    def send_servo(self, pan, tilt):
        if self.ser and self.ser.is_open:
            # Gửi chuỗi theo đúng cấu trúc ESP32 đang đợi bóc tách
            cmd = f"P:{int(pan)},T:{int(tilt)}\n"
            try:
                self.ser.write(cmd.encode())
            except:
                pass

# ==============================================================================
# CLASS 2: BỘ ĐIỀU KHIỂN TRACKING PID 
# ==============================================================================
class PanTiltTracker:
    def __init__(self):
        self.pan = 90
        self.tilt = 90
        self.last_cx = 480
        self.last_cy = 270
        self.KP = 0.25
        self.MAX_STEP = 4

    def update(self, cx, cy, frame_w=960, frame_h=540):
        center_x = frame_w // 2
        center_y = frame_h // 2

        error_x = cx - center_x
        error_y = cy - center_y

        if abs(error_x) < 15: error_x = 0
        if abs(error_y) < 15: error_y = 0

        delta_pan = (error_x * (62 / frame_w)) * self.KP
        delta_tilt = (error_y * (48 / frame_h)) * self.KP

        delta_pan = np.clip(delta_pan, -self.MAX_STEP, self.MAX_STEP)
        delta_tilt = np.clip(delta_tilt, -self.MAX_STEP, self.MAX_STEP)

        self.pan -= delta_pan
        self.tilt += delta_tilt

        # Cho phép dải góc quét chạy rộng (-10 đến 200) để test tính năng hú còi khi vượt biên 0 và 180
        self.pan = np.clip(self.pan, -10, 200) 
        self.tilt = np.clip(self.tilt, 0, 180)

        return int(self.pan), int(self.tilt)

# Khởi tạo đối tượng hệ thống
model = YOLO("best.pt")
tracker = PanTiltTracker()
serial_mgr = SerialManager(port="COM5", baudrate=115200)

# Khởi tạo Camera (Thiết lập index 1 theo cấu hình phần cứng của bạn)
cap = cv2.VideoCapture(1)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

class_names = {0: "Helicopter", 1: "Jet", 2: "Rocket"}
system_state = {"distance": 8190, "status": "IDLE", "pan": 90, "tilt": 90, "label": "-", "log_msg": "", "is_alarm": False}

def generate_frames():
    global system_state
    while True:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.03)
            continue

        frame = cv2.resize(frame, (960, 540))
        
        # Lấy dữ liệu cảm biến thời gian thực từ ESP32 gửi lên
        distance = serial_mgr.distance if (serial_mgr and serial_mgr.ser) else 8190
        motion = serial_mgr.motion if (serial_mgr and serial_mgr.ser) else 0

        status = "IDLE"
        label = "-"
        pan, tilt = tracker.pan, tracker.tilt
        target_found = False

        # ==============================================================================
        # 🚨 ĐOẠN ĐIỀU KIỆN KIỂM TRA KHOẢNG CÁCH 8000mm HOẶC CÓ CHUYỂN ĐỘNG
        # ==============================================================================
        if distance <= 8000 or motion == 1:
            status = "SCANNING"  # Vật thể ở phạm vi kích hoạt, bật trạng thái Quét YOLO
            
            results = model(frame, conf=0.45, verbose=False)
            for result in results:
                for box in result.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                    label = class_names.get(int(box.cls), "Target")
                    
                    target_found = True
                    status = "TRACKING"
                    pan, tilt = tracker.update(cx, cy, 960, 540)
                    
                    # Gửi tọa độ góc xuống ESP32 liên tục khi đang bám mục tiêu
                    serial_mgr.send_servo(pan, tilt)

                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)
                    break
                if target_found: break
        else:
            status = "OUT OF RANGE"  # Mục tiêu ở quá xa, tắt YOLO để giảm tải cho CPU

        # Nếu không tìm thấy mục tiêu (Dù đang quét hoặc ngoài phạm vi), vẫn duy trì cập nhật góc hiện tại
        if not target_found:
            serial_mgr.send_servo(pan, tilt)

        current_time = time.strftime("%H:%M:%S", time.localtime())
        is_alarm = False

        # Kiểm tra điều kiện biên để đưa cảnh báo lên giao diện Web Monitor
        if int(pan) > 180 or int(pan) <= 0:
            log_msg = f"[{current_time}] ⚠️ [CẢNH BÁO NGUY HIỂM] Máy Bay Vào Không Phận! (Góc hiện tại: {int(pan)}°)"
            is_alarm = True
        elif status == "TRACKING":
            log_msg = f"[{current_time}] [CẢNH BÁO] Phát hiện mục tiêu! Khoảng cách: {distance}mm. X={int(pan)}°, Y={int(tilt)}°"
        elif status == "SCANNING":
            log_msg = f"[{current_time}] [HỆ THỐNG] Vật thể xuất hiện trong tầm ({distance}mm)! Đang chạy phân tích thực thể..."
        else:
            log_msg = f"[{current_time}] [HỆ THỐNG] An toàn. Không phát hiện vật thể xâm nhập gần (Khoảng cách hiện tại: {distance}mm)."

        system_state = {
            "distance": int(distance),
            "status": status,
            "pan": int(pan),
            "tilt": int(tilt),
            "label": label,
            "log_msg": log_msg,
            "is_alarm": is_alarm
        }

        ret, buffer = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

@app.route('/')
def index(): return render_template_string(HTML_LAYOUT)

@app.route('/video_feed')
def video_feed(): return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/telemetry')
def telemetry():
    def stream():
        while True:
            yield f"data: {json.dumps(system_state)}\n\n"
            time.sleep(0.05)
    return Response(stream(), mimetype='text/event-stream')

# ==============================================================================
# GIAO DIỆN MONITORING DASHBOARD (HTML/CSS/JS)
# ==============================================================================
HTML_LAYOUT = """
<!DOCTYPE html>
<html lang="vi">
<head>
    <meta charset="UTF-8">
    <title>RADAR MONITORING DASHBOARD</title>
    <style>
        * { box-sizing: border-box; }
        body { 
            background: #020911; color: #00f0ff; font-family: 'Courier New', monospace; 
            margin: 0; padding: 15px; display: flex; flex-direction: column; gap: 15px;
        }
        .main-container { display: grid; grid-template-columns: 1.2fr 0.8fr; gap: 15px; }
        .panel-box { border: 1px solid #004d61; background: rgba(2, 12, 22, 0.9); padding: 12px; position: relative; }
        .panel-header { font-size: 11px; color: #61879f; border-bottom: 1px solid #002d3d; padding-bottom: 5px; margin-bottom: 10px; text-transform: uppercase; letter-spacing: 1px; }
        .cam-container { width: 100%; background: #000; border: 1px solid #002233; aspect-ratio: 16 / 9; overflow: hidden; }
        .cam-container img { width: 100%; height: 100%; object-fit: contain; }
        .radar-right-panel { display: flex; flex-direction: column; gap: 15px; }
        .radar-center-wrapper { display: flex; justify-content: center; align-items: center; padding: 10px 0; min-height: 240px; }
        canvas { background: #01060c; border-radius: 50%; box-shadow: inset 0 0 15px rgba(0, 240, 255, 0.15); }
        .bottom-dashboard { display: grid; grid-template-columns: repeat(2, 1fr); gap: 10px; }
        .telemetry-card { background: rgba(3, 20, 34, 0.7); border: 1px solid #002d3d; border-left: 4px solid #00f0ff; padding: 10px 12px; }
        .card-label { font-size: 10px; color: #61879f; margin-bottom: 4px; text-transform: uppercase; }
        .card-data { font-size: 20px; font-weight: bold; color: #ffffff; }
        .card-data span { font-size: 12px; color: #00f0ff; margin-left: 3px; }
        .terminal-log-panel { border: 1px solid #004d61; background: #00050a; padding: 10px; height: 120px; overflow-y: auto; border-radius: 4px; }
        .log-line { font-size: 12px; color: #00ff66; margin-bottom: 4px; line-height: 1.4; }
        .log-line.alarm-active { color: #ff3333 !important; font-weight: bold; animation: blinker 1s linear infinite; }
        @keyframes blinker { 50% { opacity: 0.5; } }
    </style>
</head>
<body>
    <div class="main-container">
        <div class="panel-box">
            <div class="panel-header">RADAR LIVE STREAM DETECTOR</div>
            <div class="cam-container"><img src="/video_feed"></div>
        </div>
        <div class="radar-right-panel">
            <div class="panel-box">
                <div class="panel-header">RADAR MICRO-WAVE SCANNER</div>
                <div class="radar-center-wrapper"><canvas id="radar" width="240" height="240"></canvas></div>
            </div>
            <div class="bottom-dashboard">
                <div class="telemetry-card">
                    <div class="card-label">KHOẢNG CÁCH (VL53L0X)</div>
                    <div class="card-data" id="lbl-distance">0<span>mm</span></div>
                </div>
                <div class="telemetry-card">
                    <div class="card-label">TRẠNG THÁI HỆ THỐNG</div>
                    <div class="card-data" id="lbl-status" style="color: #ffaa00; font-size:16px;">TÌM KIẾM</div>
                </div>
                <div class="telemetry-card">
                    <div class="card-label">SERVO PAN</div>
                    <div class="card-data" id="lbl-pan">0°</div>
                </div>
                <div class="telemetry-card">
                    <div class="card-label">SERVO TILT</div>
                    <div class="card-data" id="lbl-tilt">0°</div>
                </div>
            </div>
        </div>
    </div>
    <div class="terminal-log-panel" id="log-box"></div>

    <script>
        const canvas = document.getElementById("radar");
        const ctx = canvas.getContext("2d");
        const cx = canvas.width / 2; const cy = canvas.height / 2; const maxR = canvas.width / 2 - 15;
        let curPan = 90; let curDist = 8190;

        function draw() {
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            ctx.strokeStyle = "rgba(0, 240, 255, 0.15)"; ctx.lineWidth = 1;
            ctx.beginPath(); ctx.arc(cx, cy, maxR * (2000/8000), 0, 2*Math.PI); ctx.stroke();
            ctx.beginPath(); ctx.arc(cx, cy, maxR * (5000/8000), 0, 2*Math.PI); ctx.stroke();
            ctx.beginPath(); ctx.arc(cx, cy, maxR, 0, 2*Math.PI); ctx.stroke();
            ctx.beginPath(); ctx.moveTo(cx - maxR, cy); ctx.lineTo(cx + maxR, cy); ctx.stroke();
            ctx.beginPath(); ctx.moveTo(cx, cy - maxR); ctx.lineTo(cx, cy + maxR); ctx.stroke();

            let angleRad = (curPan * Math.PI) / 180;
            ctx.strokeStyle = "rgba(0, 240, 255, 0.6)"; ctx.lineWidth = 1.5;
            ctx.beginPath(); ctx.moveTo(cx, cy); ctx.lineTo(cx + maxR * Math.cos(angleRad), cy - maxR * Math.sin(angleRad)); ctx.stroke();

            if (curDist > 0 && curDist <= 8000) {
                let rPixel = maxR * (curDist / 8000);
                let targetX = cx + rPixel * Math.cos(angleRad); let targetY = cy - rPixel * Math.sin(angleRad);
                ctx.fillStyle = "#ff0000"; ctx.shadowBlur = 15; ctx.shadowColor = "#ff0000";
                ctx.beginPath(); ctx.arc(targetX, targetY, 7, 0, 2 * Math.PI); ctx.fill();
                ctx.fillStyle = "#ffffff"; ctx.shadowBlur = 0;
                ctx.beginPath(); ctx.arc(targetX, targetY, 2.5, 0, 2 * Math.PI); ctx.fill();
            }
            requestAnimationFrame(draw);
        }
        draw();

        const sse = new EventSource("/telemetry");
        const logBox = document.getElementById("log-box");
        let lastLogMsg = "";

        sse.onmessage = function(e) {
            const data = JSON.parse(e.data);
            curDist = data.distance; curPan = data.pan;
            document.getElementById("lbl-distance").innerHTML = data.distance >= 8190 ? "MAX<span>mm</span>" : `${data.distance}<span>mm</span>`;
            document.getElementById("lbl-pan").innerText = `${data.pan}°`;
            document.getElementById("lbl-tilt").innerText = `${data.tilt}°`;

            const statusContainer = document.getElementById("lbl-status");
            if (data.is_alarm) {
                statusContainer.innerText = "XÂM NHẬP BIÊN!"; statusContainer.style.color = "#ff3333";
            } else if (data.status === "TRACKING") {
                statusContainer.innerText = `BÁM: ${data.label.toUpperCase()}`; statusContainer.style.color = "#ffaa00";
            } else if (data.status === "SCANNING") {
                statusContainer.innerText = "PHÂN TÍCH AI..."; statusContainer.style.color = "#00ff66";
            } else {
                statusContainer.innerText = "CHỜ MỤC TIÊU"; statusContainer.style.color = "#61879f";
            }

            if (data.log_msg && data.log_msg !== lastLogMsg) {
                lastLogMsg = data.log_msg;
                const p = document.createElement("div");
                p.className = data.is_alarm ? "log-line alarm-active" : "log-line";
                p.innerText = data.log_msg;
                logBox.appendChild(p); logBox.scrollTop = logBox.scrollHeight;
            }
        };
    </script>
</body>
</html>
"""

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5500, debug=False)
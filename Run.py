from ultralytics import YOLO
import cv2
import time

# ==========================
# LOAD MODEL
# ==========================
model = YOLO("best.pt")

# ==========================
# QUÉT CAMERA
# ==========================
print("===== DANH SÁCH CAMERA =====")

available_cams = []

for i in range(10):
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        print(f"[{i}] Camera khả dụng")
        available_cams.append(i)       
        cap.release()

if len(available_cams) == 0:
    print("Không tìm thấy camera!")
    exit()

cam_id = int(input("Nhập ID camera muốn dùng: "))

# ==========================
# MỞ CAMERA ĐÃ CHỌN
# ==========================
cap = cv2.VideoCapture(cam_id)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

prev_time = time.time()

while True:
    ret, frame = cap.read()

    if not ret:
        break

    results = model.predict(
        frame,
        conf=0.5,
        imgsz=640,
        verbose=False
    )

    frame = results[0].plot()

    current_time = time.time()
    fps = 1 / (current_time - prev_time)
    prev_time = current_time

    cv2.putText(
        frame,
        f"FPS: {int(fps)}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    cv2.imshow("Air Defense System", frame)

    key = cv2.waitKey(1)

    if key == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()
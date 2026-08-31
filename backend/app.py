import base64
import cv2
import numpy as np
from flask import Flask, jsonify, request
from flask_cors import CORS

app = Flask(__name__)

# เปิดใช้งาน CORS เพื่อให้ Frontend จากต่างเครื่อง/ต่าง Port เรียกใช้ API ได้
CORS(app)


# 1. Health Check Endpoint
@app.route("/health", methods=["GET"])
def health_check():
    return jsonify({"status": "ok", "message": "Server is running"}), 200


# 2. Image Processing Endpoint
@app.route("/process-image", methods=["POST"])
def process_image():
    # ตรวจสอบว่ามีการส่งไฟล์ภาพมาใน request หรือไม่
    if "image" not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    try:
        # อ่านไฟล์ภาพจาก Request เข้า Memory
        file_bytes = np.frombuffer(file.read(), np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        if img is None:
            return jsonify({"error": "Invalid image file"}), 400

        # อ่านค่า Threshold สำหรับ Canny Edge (ถ้าไม่มีให้ใช้ค่า Default 100, 200)
        threshold1 = int(request.form.get("threshold1", 100))
        threshold2 = int(request.form.get("threshold2", 200))

        # --- Image Processing: Canny Edge Detection ---
        # 1. แปลงภาพเป็น Grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # 2. ทำ Gaussian Blur เพื่อลด Noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        # 3. คำนวณ Canny Edge
        edges = cv2.Canny(blurred, threshold1, threshold2)

        # --- Base64 Encoding ---
        # แปลงภาพ Numpy Array กลับเป็น Image Buffer (.png)
        _, buffer = cv2.imencode(".png", edges)
        # เข้ารหัสเป็น Base64 string
        base64_image = base64.b64encode(buffer).decode("utf-8")
        data_url = f"data:image/png;base64,{base64_image}"

        # ส่ง Response กลับเป็น JSON
        return (
            jsonify(
                {
                    "status": "success",
                    "algorithm": "Canny Edge Detection",
                    "result_image": data_url,
                }
            ),
            200,
        )

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    # กำหนด host='0.0.0.0' เพื่อให้เครื่องอื่นในวง LAN สามารถเชื่อมต่อเข้ามาได้
    app.run(host="0.0.0.0", port=5000, debug=True)
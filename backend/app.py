import base64
import cv2
import numpy as np
from flask import Flask, jsonify, request
from flask_cors import CORS

app = Flask(__name__)
# เปิด CORS เพื่อให้ Frontend เรียกใช้งานได้ข้าม Origin/IP
CORS(app)


# 1. Health Check Endpoint
@app.route("/health", methods=["GET"])
def health_check():
    return jsonify({"status": "ok", "message": "Backend server is running"}), 200


# 2. Main Image Processing Endpoint matching JS `/api/process`
@app.route("/api/process", methods=["POST"])
def process_image():
    # ตรวจสอบการส่งไฟล์ภาพ
    if "image" not in request.files:
        return (
            jsonify({"success": False, "error": "No image file provided"}),
            400,
        )

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"success": False, "error": "Empty image file"}), 400

    try:
        # อ่านไฟล์ภาพจาก Memory เข้าสู่ OpenCV
        file_bytes = np.frombuffer(file.read(), np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        if img is None:
            return (
                jsonify({"success": False, "error": "Invalid image format"}),
                400,
            )

        # รับค่า operation จาก Frontend
        operation = request.form.get("operation", "canny")

        # --- Algorithms Processing ---
        if operation == "canny":
            # อ่านค่า threshold1 และ threshold2 (default: 100, 200)
            t1 = int(request.form.get("threshold1", 100))
            t2 = int(request.form.get("threshold2", 200))

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            processed_img = cv2.Canny(blurred, t1, t2)

        elif operation == "sobel":
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            # คำนวณ Gradient Magnitude
            magnitude = cv2.magnitude(sobelx, sobely)
            processed_img = cv2.normalize(
                magnitude, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U
            )

        elif operation == "harris":
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            gray_float = np.float32(gray)
            dst = cv2.cornerHarris(gray_float, 2, 3, 0.04)
            dst = cv2.dilate(dst, None)
            processed_img = img.copy()
            # วาดจุดมุมด้วยสีแดง
            processed_img[dst > 0.01 * dst.max()] = [0, 0, 255]

        elif operation == "grayscale":
            processed_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        elif operation == "blur":
            processed_img = cv2.GaussianBlur(img, (15, 15), 0)

        else:
            return (
                jsonify(
                    {"success": False, "error": f"Unknown operation: {operation}"}
                ),
                400,
            )

        # --- Encode Image Result to Base64 ---
        _, buffer = cv2.imencode(".png", processed_img)
        base64_str = base64.b64encode(buffer).decode("utf-8")
        data_url = f"data:image/png;base64,{base64_str}"

        # ส่ง Response กลับโครงสร้างตรงกับที่ JS ฝั่ง Frontend รอรับ
        return (
            jsonify(
                {
                    "success": True,
                    "operation": operation,
                    "image": data_url,
                }
            ),
            200,
        )

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


if __name__ == "__main__":
    # รันบน 0.0.0.0 Port 5000 ให้เครื่องอื่นเข้าถึงได้
    app.run(host="0.0.0.0", port=5000, debug=True)
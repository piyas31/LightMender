import os
import cv2
import numpy as np
from flask import Flask, render_template, request, send_from_directory, jsonify

app = Flask(__name__)

UPLOAD_FOLDER = 'static/uploads'
PROCESSED_FOLDER = 'static/processed'

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['PROCESSED_FOLDER'] = PROCESSED_FOLDER

# -------- Image Processing Function ----------
def enhance_image(input_path, output_path, smooth=3, sharp=1.8, brightness=30, contrast=1.2):
    img = cv2.imread(input_path)

    # Convert to YCrCb color space for better brightness control
    img_ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)

    # Split channels
    y, cr, cb = cv2.split(img_ycrcb)

    # Apply histogram equalization on the Y channel (luminance)
    y_eq = cv2.equalizeHist(y)

    # Merge channels back
    img_ycrcb_eq = cv2.merge([y_eq, cr, cb])
    img_eq = cv2.cvtColor(img_ycrcb_eq, cv2.COLOR_YCrCb2BGR)

    # Smoothness (Gaussian Blur)
    if smooth > 0:
        img_eq = cv2.GaussianBlur(img_eq, (smooth*2+1, smooth*2+1), 0)

    # Sharpness
    if sharp != 1.0:
        blur = cv2.GaussianBlur(img_eq, (0, 0), 3)
        img_eq = cv2.addWeighted(img_eq, sharp, blur, -(sharp-1), 0)

    # Brightness & Contrast adjustment
    img_eq = cv2.convertScaleAbs(img_eq, alpha=contrast, beta=brightness)

    # Gray level thresholding for low-light areas
    gray = cv2.cvtColor(img_eq, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 60, 255, cv2.THRESH_BINARY)  # threshold=60 adjust করতে পারেন

    # Mask low light regions
    mask = cv2.bitwise_not(thresh)

    # Enhance low light regions separately (boost brightness and contrast)
    low_light_enhanced = cv2.convertScaleAbs(img_eq, alpha=1.5, beta=40)

    # Combine normal and enhanced parts using mask
    final = cv2.bitwise_and(img_eq, img_eq, mask=thresh) + cv2.bitwise_and(low_light_enhanced, low_light_enhanced, mask=mask)

    cv2.imwrite(output_path, final)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files['photo']
        if file.filename == '':
            return "No file", 400
        
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)

        output_filename = "processed_" + file.filename
        output_path = os.path.join(app.config['PROCESSED_FOLDER'], output_filename)

        enhance_image(filepath, output_path)

        return render_template("index.html",
                               uploaded_image=filepath,
                               processed_image=output_path,
                               download_filename=output_filename,
                               filename=file.filename)

    return render_template("index.html", uploaded_image=None, processed_image=None)

@app.route("/process", methods=["POST"])
def process_image():
    data = request.json
    filename = data.get("filename")
    smooth = int(data.get("smooth", 3))
    sharp = float(data.get("sharp", 1.8))
    brightness = int(data.get("brightness", 30))
    contrast = float(data.get("contrast", 1.2))

    input_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    output_filename = "processed_" + filename
    output_path = os.path.join(app.config['PROCESSED_FOLDER'], output_filename)

    enhance_image(input_path, output_path, smooth, sharp, brightness, contrast)

    return jsonify({"processed_image": "/" + output_path})

@app.route('/download/<filename>')
def download_file(filename):
    return send_from_directory(app.config['PROCESSED_FOLDER'], filename, as_attachment=True)

if __name__ == "__main__":
    app.run(debug=True)

import os
import gdown
import cv2
import numpy as np
from flask import Flask, render_template, Response, request, jsonify
from cv2 import dnn
from werkzeug.utils import secure_filename
from threading import Thread
import time
from PIL import Image
from io import BytesIO
import base64

app = Flask(__name__)

# Paths
UPLOAD_FOLDER = 'static/uploads/'
OUTPUT_FOLDER = 'static/output/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER

# Ensure necessary folders exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Google Drive file IDs
PROTO_FILE_ID = 'your_prototxt_file_id'  # Replace with actual file ID
MODEL_FILE_ID = 'your_caffemodel_file_id'  # Replace with actual file ID
PTS_FILE_ID = 'your_pts_in_hull_file_id'  # Replace with actual file ID

# Paths to save the downloaded models
proto_file = 'static/models/colorization.prototxt'
model_file = 'static/models/chromify_mod.caffemodel'
hull_pts = 'static/models/pts_in_hull.npy'

# Function to download files from Google Drive
def download_from_drive(file_id, destination):
    url = f'https://drive.google.com/uc?id={file_id}'
    gdown.download(url, destination, quiet=False)

# Download models from Google Drive if not already present
if not os.path.exists(proto_file):
    download_from_drive(PROTO_FILE_ID, proto_file)

if not os.path.exists(model_file):
    download_from_drive(MODEL_FILE_ID, model_file)

if not os.path.exists(hull_pts):
    download_from_drive(PTS_FILE_ID, hull_pts)

# Load pre-trained model
net = dnn.readNetFromCaffe(proto_file, model_file)
kernel = np.load(hull_pts)
class8 = net.getLayerId("class8_ab")
conv8 = net.getLayerId("conv8_313_rh")
pts = kernel.transpose().reshape(2, 313, 1, 1)
net.getLayer(class8).blobs = [pts.astype("float32")]
net.getLayer(conv8).blobs = [np.full([1, 313], 2.606, dtype="float32")]

# Webcam setup
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not access the webcam.")

def colorize_frame(frame):
    try:
        # Resize the frame for the model (224x224)
        resized_frame = cv2.resize(frame, (224, 224)).astype("float32") / 255.0
        lab_img = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2LAB)
        L = cv2.split(lab_img)[0]  # Luminance channel
        L -= 50  # Center L-channel

        net.setInput(cv2.dnn.blobFromImage(L))  # Run model for colorization
        ab_channel = net.forward()[0, :, :, :].transpose((1, 2, 0))
        ab_channel = cv2.resize(ab_channel, (frame.shape[1], frame.shape[0]))  # Resize to original frame size
        # Reconstruct the LAB image
        original_lab_L = cv2.cvtColor(frame.astype("float32") / 255.0, cv2.COLOR_BGR2LAB)
        L_original = cv2.split(original_lab_L)[0]
        colorized = np.concatenate((L_original[:, :, np.newaxis], ab_channel), axis=2)
        colorized = cv2.cvtColor(colorized, cv2.COLOR_LAB2BGR)
        colorized = np.clip(colorized, 0, 1)
        return (255 * colorized).astype("uint8")  # Convert back to 8-bit color
    except Exception as e:
        print(f"Error during colorization: {e}")
        return frame  # Return the original frame in case of error

@app.route('/')
def homepage():
    return render_template('homepage.html')

@app.route('/webcam')
def webcam():
    return render_template('webcam.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

def generate():
    """Stream frames from the webcam with real-time colorization."""
    while True:
        ret, frame = cap.read()  # Capture frame from webcam
        if not ret:
            print("Error: Could not read frame.")
            break

        # Apply colorization to the captured frame
        colorized_frame = colorize_frame(frame)

        # Encode the colorized frame into JPEG format for streaming
        ret, jpeg = cv2.imencode('.jpg', colorized_frame)
        if not ret:
            print("Error: Could not encode frame.")
            break

        # Yield the frame as a response in a multipart format
        frame_response = jpeg.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_response + b'\r\n')

@app.route('/capture')
def capture_image():
    ret, frame = cap.read()  # Capture frame from webcam
    if ret:
        # Save the captured frame
        timestamp = int(time.time())
        image_path = f'static/output/captured_image_{timestamp}.jpg'
        cv2.imwrite(image_path, frame)

        # Colorize the captured image
        colorized_frame = colorize_frame(frame)

        # Save the colorized image
        colorized_image_path = f'static/output/colorized_image_{timestamp}.jpg'
        cv2.imwrite(colorized_image_path, colorized_frame)

        # Return the colorized image URL
        return jsonify({'image_url': f'/{colorized_image_path}'}), 200
    
    return jsonify({'error': 'Failed to capture image'}), 500

@app.route('/upload', methods=['GET', 'POST'])
def upload_image():
    if request.method == 'POST':
        if 'image' not in request.files or request.files['image'].filename == '':
            return render_template('upload.html', colorized_image=None, error="No file selected.")

        file = request.files['image']
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)

            uploaded_img = cv2.imread(file_path)
            colorized_img = colorize_frame(uploaded_img)

            colorized_filename = 'colorized_' + filename
            colorized_img_path = os.path.join(app.config['OUTPUT_FOLDER'], colorized_filename)
            cv2.imwrite(colorized_img_path, colorized_img)

            colorized_image_url = os.path.join('static', 'output', colorized_filename)
            return render_template('upload.html', colorized_image=colorized_image_url)

    return render_template('upload.html', colorized_image=None)

def allowed_file(filename):
    allowed_extensions = {'png', 'jpg', 'jpeg', 'gif'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_extensions

@app.route('/shutdown', methods=['POST'])
def shutdown():
    global cap
    if cap.isOpened():
        cap.release()
    cv2.destroyAllWindows()
    return "Webcam and resources released", 200

if __name__ == '__main__':
    app.run(debug=True)

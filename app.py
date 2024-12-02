from flask import Flask, render_template, Response, request, jsonify
import cv2
import os
import numpy as np
from cv2 import dnn
from werkzeug.utils import secure_filename
from threading import Thread
import time

app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = 'static/uploads/'
OUTPUT_FOLDER = 'static/output/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER

# Ensure necessary folders exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Model paths
proto_file = r"C:\Users\CLienT\Documents\imagecolorization\model\colorization.prototxt"
model_file = r"C:\Users\CLienT\Documents\imagecolorization\model\chromify_mod.caffemodel"
hull_pts = r"C:\Users\CLienT\Documents\imagecolorization\model\pts_in_hull.npy"

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
else:
    print("Webcam accessed successfully.")

def colorize_frame(frame):
    """Colorize a single frame."""
    try:
        # Resize frame and normalize to [0, 1]
        resized_frame = cv2.resize(frame, (224, 224)).astype("float32") / 255.0
        
        # Convert frame to LAB color space (L is the luminance channel, A and B are chrominance channels)
        lab_img = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2LAB)
        L = cv2.split(lab_img)[0]  # Extract L channel
        L -= 50  # Normalize L channel by subtracting 50

        # Set input to the neural network and get output (AB channels)
        net.setInput(cv2.dnn.blobFromImage(L))  # Pass L channel to the network
        ab_channel = net.forward()[0, :, :, :].transpose((1, 2, 0))  # Get AB channels
        ab_channel = cv2.resize(ab_channel, (frame.shape[1], frame.shape[0]))  # Resize AB to match the original frame size

        # Convert original frame to LAB, extract the L channel
        original_lab_L = cv2.cvtColor(frame.astype("float32") / 255.0, cv2.COLOR_BGR2LAB)
        L_original = cv2.split(original_lab_L)[0]

        # Combine original L channel with the predicted AB channels
        colorized = np.concatenate((L_original[:, :, np.newaxis], ab_channel), axis=2)
        
        # Convert back to BGR color space
        colorized = cv2.cvtColor(colorized, cv2.COLOR_LAB2BGR)
        
        # Ensure the colorized image is in the valid range [0, 1]
        colorized = np.clip(colorized, 0, 1)

        # Convert to 8-bit format (0-255) for display or saving
        return (255 * colorized).astype("uint8")
        colorized_frame = frame
    except Exception as e:
        print(f"Error during colorization: {e}")
        return frame  # Return the original frame in case of an error

# Folder to store captured images
image_folder = r'C:\Users\CLienT\Documents\imagecolorization\captured_images'
if not os.path.exists(image_folder):
    os.makedirs(image_folder)

@app.route('/')
def homepage():
    return render_template('homepage.html')

@app.route('/webcam')
def webcam():
    return render_template('webcam.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/capture')
def capture_image():
    ret, frame = cap.read()  # Capture frame from webcam
    if ret:
        # Save the raw captured frame as an image file without any colorization
        timestamp = int(time.time())  # Use the time module to get the timestamp
        image_path = 'static/captured_image.jpg'
        cv2.imwrite(image_path, frame)  # Save the captured raw image
        
        # Return the relative path of the captured image to be used on the frontend
        return jsonify({'image_url': image_path})
    
    return jsonify({'error': 'Failed to capture image'}), 500

@app.route('/colorize', methods=['POST'])
def colorize_image():
    data = request.get_json()
    image_url = data.get('image_url')

    if image_url:
        # Simulate the colorization process (replace this with your actual model logic)
        image_path = image_url.split('static/')[-1]  # Extract the filename from the URL
        original_image_path = f'static/{image_path}'
        colorized_image_path = 'static/colorized_image.jpg'

        # Load the captured image
        frame = cv2.imread(original_image_path)

        # Simulate colorization (replace with actual colorization model)
        colorized_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Dummy transformation

        # Save the colorized image
        cv2.imwrite(colorized_image_path, colorized_frame)

        # Return the path to the colorized image
        return jsonify({'colorized_image_url': colorized_image_path})

    return jsonify({'error': 'Failed to colorize image'}), 400


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

        
import base64
from io import BytesIO
from PIL import Image

@app.route('/process_image', methods=['POST'])
def process_image():
    try:
        data = request.get_json()  # Receive the JSON data from the frontend
        image_data = data['image']
        
        # Decode the base64 image data
        image_data = image_data.split(",")[1]  # Remove the prefix
        image = Image.open(BytesIO(base64.b64decode(image_data)))

        # Convert image to OpenCV format for colorization
        image = np.array(image)
        colorized_image = colorize_frame(image)  # Apply colorization

        # Save the colorized image
        colorized_image_path = os.path.join(app.config['OUTPUT_FOLDER'], 'colorized_image.jpg')
        cv2.imwrite(colorized_image_path, colorized_image)

        # Return the path to the colorized image
        return jsonify({"colorized_image": '/static/output/colorized_image.jpg'})
    except Exception as e:
        print(f"Error during image processing: {e}")
        return jsonify({"error": "Error processing image"}), 500

def allowed_file(filename):
    allowed_extensions = {'png', 'jpg', 'jpeg', 'gif'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_extensions

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

def allowed_video_file(filename):
    ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def colorize_video(input_video_path):
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return None

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    base_filename = os.path.splitext(os.path.basename(input_video_path))[0]
    output_video_path = os.path.join(OUTPUT_FOLDER, f"{base_filename}-colorized.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        colorized_frame = colorize_frame(frame)
        out.write(colorized_frame)

    cap.release()
    out.release()
    return output_video_path  # Ensure this is returned


@app.route('/upload_video', methods=['GET', 'POST'])
def upload_video():
    if request.method == 'POST':
        if 'video' not in request.files or request.files['video'].filename == '':
            return render_template('upload_video.html', error="No video file selected.")

        video = request.files['video']
        input_filename = secure_filename(video.filename)
        input_path = os.path.join(app.config['UPLOAD_FOLDER'], input_filename)
        video.save(input_path)

        colorized_video_path = colorize_video(input_path)  # Get the output path from colorize_video function
        if colorized_video_path:
            # Construct the relative path for the colorized video to be used in the template
            colorized_video_url = os.path.join('static', 'output', os.path.basename(colorized_video_path))
            return render_template('upload_video.html', colorized_video=colorized_video_url)
        else:
            return render_template('upload_video.html', error="Error processing video.")

    return render_template('upload_video.html')


if __name__ == '__main__':
    app.run(debug=True)

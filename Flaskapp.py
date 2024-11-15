import os
import cv2
import json
import time
import base64
from flask import Flask, jsonify, render_template, Response
from flask_socketio import SocketIO
from ultralytics import YOLO

app = Flask(__name__)
socketio = SocketIO(app)

model = YOLO('best.pt')  # Load your trained YOLO model
THRESHOLD = 0.5  # Confidence threshold
ALERT_REGION = [(100, 100), (500, 500)]  # Define alert region (coordinates)

# Ensure the "suspicious_captures" folder exists
output_folder = "suspicious_captures"
os.makedirs(output_folder, exist_ok=True)

def is_within_alert_region(x, y):
    """Check if the detection is within the alert region."""
    (x1, y1), (x2, y2) = ALERT_REGION
    return x1 <= x <= x2 and y1 <= y <= y2

def generate_frames():
    cap = cv2.VideoCapture(0)  # Open camera
    if not cap.isOpened():
        print("Error: Unable to access the camera.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Run the YOLO model on the frame
        results = model(frame)[0]
        
        # Flag to track if an alert is triggered
        alert_triggered = False
        alert_details = {}

        # Process detections and draw bounding boxes
        for result in results.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = result
            if score > THRESHOLD and is_within_alert_region(int(x1), int(y1)):
                activity_name = model.names[int(class_id)]
                alert_triggered = True
                
                # Draw bounding box and label on the frame
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                cv2.putText(frame, f"{activity_name} ({score:.2f})", (int(x1), int(y1) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                # Prepare alert details to send to the frontend
                timestamp = time.strftime('%Y%m%d_%H%M%S')
                alert_details = {
                    "activity": activity_name,
                    "confidence": round(score, 2),
                    "coordinates": (x1, y1, x2, y2),
                    "timestamp": timestamp
                }

                # Save the frame as an image in the suspicious_captures folder
                image_filename = f"{output_folder}/{activity_name}_{timestamp}.jpg"
                cv2.imwrite(image_filename, frame)

                # Convert the captured image to base64 for sending to the frontend
                _, buffer = cv2.imencode('.jpg', frame)
                frame_base64 = base64.b64encode(buffer).decode('utf-8')
                alert_details["image"] = frame_base64

                # Emit alert to frontend via SocketIO
                socketio.emit('alert', alert_details)

                # Log the alert details
                with open('alert_log.json', 'a') as log_file:
                    json.dump(alert_details, log_file)
                    log_file.write('\n')

                break  # Stop after first detected alert per frame

        # Encode the processed frame for streaming
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()

@app.route('/video_feed')
def video_feed():
    """Stream processed video feed with bounding boxes."""
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index():
    return render_template('index4.html')

if __name__ == "__main__":
    socketio.run(app, debug=True)

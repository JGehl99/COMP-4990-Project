from flask import Flask, render_template, Response, request
import cv2

# setup
app = Flask(__name__)
camera = cv2.VideoCapture(0)

# capture video feed
def gen_frames():
    while True:
        success, frame = camera.read()
        
        if not success:
            break
        else:

            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


# render html            
@app.route('/')
def index():
    return render_template('index.html')

# render feed
@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# File upload
@app.route('/upload', methods=['POST'])
def file_upload():
    img1 = request.files['img1']
    img2 = request.files['img2']
    # TODO Run SIFT here with these files




#run
if __name__ == "__main__":
    app.run(debug=True)
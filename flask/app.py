import numpy
from flask import Flask, render_template, request
import cv2

# setup
app = Flask(__name__)


# render html            
@app.route('/')
def index():
    return render_template('index.html')


# File upload
@app.route('/upload', methods=['POST'])
def file_upload():

    if request.method == 'POST':
        img1 = img_upload_to_cv2(request.files['img1'].read())
        img2 = img_upload_to_cv2(request.files['img2'].read())
        tolerance = request.form.get('tol')

        print(tolerance)

        # sift_setup(img1, img2, tolerance)

    return render_template('index.html')


def img_upload_to_cv2(image_str):
    return cv2.imdecode(numpy.fromstring(image_str, numpy.uint8), cv2.IMREAD_UNCHANGED)


app.run(debug=True)

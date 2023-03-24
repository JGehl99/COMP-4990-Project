import os

import numpy
from flask import Flask, render_template, request, session, redirect, url_for
import cv2
from sift import sift_setup
import uuid

# setup
app = Flask(__name__)

app.secret_key = 'jYxz/MBS&CXNHc.Gb6/WR^b[s/%fNLG'


# render html            
@app.route('/')
def index():

    # If user hasn't been on the site before, assign them a UUID
    if 'UUID' not in session:
        session['UUID'] = uuid.uuid4().hex

    return render_template('index.html')


# File upload
@app.route('/upload', methods=['POST'])
def file_upload():

    # If post method
    if request.method == 'POST':

        # Get images and min kp from form
        img1 = img_upload_to_cv2(request.files['img1'].read())
        img2 = img_upload_to_cv2(request.files['img2'].read())
        kp = float(request.form.get('kp'))

        # Get result, time, and image from sift function
        result, time, image = sift_setup(img1, img2, kp)

        # Save image to static/images folder
        cv2.imwrite(os.path.join(os.getcwd(), 'flask', 'static', 'images', (session['UUID'] + '.jpg')), image)

        # Prepare data for jinja template
        data = [result, time, url_for('static', filename='images/' + session['UUID'] + '.jpg')]

        # Render the template
        return render_template('results.html', data=data)

    else:
        return redirect(url_for('index'))


# Helper function to convert images
def img_upload_to_cv2(image_str):
    return cv2.imdecode(numpy.fromstring(image_str, numpy.uint8), cv2.IMREAD_UNCHANGED)


app.run(debug=True)

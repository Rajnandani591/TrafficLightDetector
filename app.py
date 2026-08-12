
from flask import Flask, render_template, request, redirect, url_for
from violations_detector import process_file
import os
from datetime import datetime

app = Flask(__name__)

UPLOAD_FOLDER = 'static/uploads'
RESULT_FOLDER = 'static/results'

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)

    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    upload_path = os.path.join(UPLOAD_FOLDER, timestamp + '_' + file.filename)
    file.save(upload_path)

    # Process image/video
    result_files, violations_info = process_file(upload_path, RESULT_FOLDER)

    return render_template('result.html', results=result_files, info=violations_info)

if __name__ == '__main__':
    app.run(debug=True)

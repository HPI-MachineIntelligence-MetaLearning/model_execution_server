from execute_model import Predictor
import yaml
from flask import Flask, request
from werkzeug.utils import secure_filename
from flask_cors import CORS
import json
import threading
import queue
import tempfile
import os

app = Flask(__name__)
CORS(app)
processed_imgs = queue.Queue()


def load_config(path):
    with open(path, 'r') as f:
        return yaml.load(f)

cfg = load_config('config.yml')
predictor = Predictor(cfg['model_path'])


def predict_images(imgs, tempdir):
    for img in imgs:
        bboxes, labels, scores = predictor.run(img)
        processed_imgs.put({'bboxes': bboxes,
                            'labels': labels,
                            'scores': scores})
    del tempdir


@app.route("/", methods=['POST'])
def start_processing():
    folder = tempfile.TemporaryDirectory()
    files = request.files.getlist('image')
    filepaths = []
    for file in files:
        filename = secure_filename(file.filename)
        filepath = os.path.join(folder.name, filename)
        file.save(filepath)
        filepaths.append(filepath)
    t = threading.Thread(target=predict_images,
                         args=(filepaths, folder))
    t.start()
    return 'Prediction started.'


@app.route("/set_output", methods=['POST'])
def set_output_folder():
    global cfg
    print(request.form)
    path = request.form['path']
    cfg['default_output_folder'] = path
    return 'Result output folder set to {}.'.format(path)


@app.route("/save_output", methods=['POST'])
def save_output():
    global cfg
    print(request.form)
    annotation = request.form['annotation']
    name = request.form['name']
    filepath = os.path.join(cfg['default_output_folder'], name)
    try:
        if not os.path.exists(cfg['default_output_folder']):
            os.makedirs(cfg['default_output_folder'])
        with open(filepath, 'w') as file:
            file.write(annotation)
    except PermissionError:
        return 'Insufficient permission to write in that folder.', 500
    return 'Annotation written to {}.'.format(filepath)


@app.route("/", methods=['GET'])
def get_img():
    global processed_imgs
    try:
        result = processed_imgs.get(timeout=2)
        return json.dumps(result)
    except queue.Empty:
        return 'No items in queue.'

if __name__ == '__main__':
    app.run()

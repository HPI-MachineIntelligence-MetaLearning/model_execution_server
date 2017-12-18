from execute_model import Predictor
import yaml
from flask import Flask, request
from flask_cors import CORS
import json

app = Flask(__name__)
CORS(app)


def load_config(path):
    with open(path, 'r') as f:
        return yaml.load(f)

cfg = load_config('config.yml')
predictor = Predictor(cfg['model_path'])


@app.route("/", methods=['POST'])
def handle_img():
    result = []
    for f in request.files.getlist('image'):
        bboxes, labels, scores = predictor.run(f)
        result.append({'bboxes': bboxes,
                       'labels': labels,
                       'scores': scores})
    return json.dumps(result)

if __name__ == '__main__':
    app.run()

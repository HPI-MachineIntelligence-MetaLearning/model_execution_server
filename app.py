import execute_model
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


@app.route("/", methods=['POST'])
def handle_img():
    f = request.files['image']
    bboxes, labels, scores = execute_model.run(f, cfg['model_path'])
    result = {'bboxes': bboxes,
              'labels': labels,
              'scores': scores}
    return json.dumps(result)

if __name__ == '__main__':
    app.run()

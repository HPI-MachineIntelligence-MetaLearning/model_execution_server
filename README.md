# model_execution_server

This is a server providing simple execution functionality to get bounding boxes from a given image.

## Installation

### General Setup

For execution, first install the enviroment using conda:
```
conda env create -f environment.yml
source activate model-execution-server
```
Then simply run the application:

```
python app.py
```

The server then runs on port 5000 (default for now, adjustments may come later)

### Configuration

You need a pretrained model in order for this to work. (Kinda obvious)
If you have access to our server, some models are in the folder `~/multi-building-detector/result`. Download one of these or train them yourself.

Then, store the model in a folder and adjust the `config.yml` to state the correct path.

## Interface

Currently, only one route is implemented:

Request method: POST

Parameters: 'image', a _File_ containing the image to be analyzed

Response: JSON, containing the bounding boxes, labels and scores (currently super nested, but maybe this is necessary, further investigation needed).
For example:
```json
{"bboxes": [[[147.03463745117188, 56.062950134277344, 332.25201416015625, 246.7890167236328]]],
"scores": [[0.734071671962738]],
"labels": [[2]]}
```

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

The server accepts requests on the root route with two methods:

Request method: POST

Parameters: 'image', one or multiple _Files_ containing the images to be analyzed

Response: 'Prediction started.' if everything went ok.

---

Request method: GET

Parameters: none

Response: In the case of success: JSON, containing the bounding boxes, labels and scores.
For example:
```json
{"bboxes": [[[147.03463745117188, 56.062950134277344, 332.25201416015625, 246.7890167236328]]],
"scores": [[0.734071671962738]],
"labels": [[2]]}
```

Else: 'No items in queue.' if either not all images have finished processing or all images have been processed and requested thereafter.

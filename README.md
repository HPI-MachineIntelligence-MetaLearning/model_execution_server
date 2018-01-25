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

You need a pretrained model in order for this to work.
If you have access to our server, some models are in the folder `~/multi-building-detector/result`. Download one of these or train them yourself.

Then, store the model in a folder and adjust the `config.yml` to state the correct path.

Additionally, this server stores the annotations it gets from the frontend. The default path where such an annotation is stored is also present in the `config.yml`, `default_output_folder`. Unless otherwise specified via the frontend (done with the request routes below), files will be saved there.

## Interface

Route: /

Request method: POST

Parameters: 'image', one or multiple _Files_ containing the images to be analyzed

Response: 'Prediction started.' if everything went ok.

---

Route: /

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


---

Route: /set_output

Request method: POST

Parameters: 'path', a string to determine in which directory the files should be saved. If no requests go to that route, files are saved in the `default_output_folder` specified in the `config.yml`.

Response: 'Result output folder set to <path>'

---

Route: /save_output

Request method: POST

Parameters: 'annotation', a string containing the annotations in a JSON format
			'name', the name of the file to be saved

Response: 'Annotation written to <path>', if everything went ok.

Else: 'Insufficient permission', if the folder specified via the `/set_route` request is not accessible for this application.

---

Route: /reset

Request method: POST

Response: 'Queue cleared.', if everything went ok. (The Queue was emptied and all running processes terminated.)


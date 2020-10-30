# Yolov3 Object Detection with Flask and Tensorflow 2.0 (APIs and Detections)

## Installation
```

#### Pip
```bash
# TensorFlow CPU
pip install -r requirements.txt

# TensorFlow GPU
pip install -r requirements-gpu.txt
```

### Downloading official pretrained weights

```
# yolov3
wget https://pjreddie.com/media/files/yolov3.weights -O weights/yolov3.weights

# yolov3-tiny
wget https://pjreddie.com/media/files/yolov3-tiny.weights -O weights/yolov3-tiny.weights
```

  
### Saving yolov3 weights as TensorFlow models.
Load the weights using `load_weights.py` script. This will convert the yolov3 weights into TensorFlow .ckpt model files!

```
# yolov3
python load_weights.py

# yolov3-tiny
python load_weights.py --weights ./weights/yolov3-tiny.weights --output ./weights/yolov3-tiny.tf --tiny
```


## Running the Flask App and Using the APIs
Now you can run a Flask application to create two object detections APIs in order to get detections through REST endpoints.
Initialize and run the Flask app on port 5000 of your local machine by running the following command from the root directory of this repo in a command prompt or shell.

```bash
python app.py
```


### Detections API (http://localhost:5000/detections)
While app.py is running the first available API is a POST routed to /detections on port 5000 of localhost. This endpoint takes in images as input and returns a JSON response with all the detections found within each image (classes found within the images and the associated confidence)

You can test out the APIs using Postman or through Curl commands (both work fine). You may have to download them if you don't already have them.

#### Accessing Detections API with Postman (RECOMMENDED)
Access the /detections API through Postman.


The response should be the output image.
!

#### Accessing Detections API with Curl 
To access and test the API through Curl, open a second command prompt or shell (may have to run as Administrator). Then cd your way to the root folder of this repository (Object-Detection-API) and run the following command.
```bash
curl.exe -X POST -F images=@data/images/dog.jpg "http://localhost:5000/detections"
```
The JSON response should be outputted to the commmand prompt if it worked successfully.

### Image API (http://localhost:5000/image)
While app.py is running the second available API is a POST routed to /image on port 5000 of localhost. This endpoint takes in a single image as input and returns a string encoded image as the response with all the detections now drawn on the image.

#### Accessing Detections API with Postman (RECOMMENDED)
Access the /image API through Postman by configuring the following.
Set method to POST and set key as image file and upload your file and send it!

The uploaded image should be returned with the detections now drawn.

#### Accessing Detections API with Curl 
To access and test the API through Curl, open a second command prompt or shell (may have to run as Administrator). Then cd your way to the root folder of this repository (Object-Detection-API) and run the following command.
```bash
curl.exe -X POST -F images=@data/images/dog.jpg "http://localhost:5000/image" --output test.png
```
This will save the returned image to the current folder as test.png (can't output the string encoded image to command prompt)


## Running just the TensorFlow model
The tensorflow model can also be run not using the APIs but through using `detect.py` script. 

Don't forget to set the IoU (Intersection over Union) and Confidence Thresholds within your yolov3-tf2/models.py file

### Usage examples
Let's run an example or two using sample images found within the data/images folder. 
```bash
# yolov3
python detect.py --images "data/images/dog.jpg, data/images/office.jpg"

# yolov3-tiny
python detect.py --weights ./weights/yolov3-tiny.tf --tiny --images "data/images/dog.jpg"

# webcam
python detect_video.py --video 0

# video file
python detect_video.py --video data/video/paris.mp4 --weights ./weights/yolov3-tiny.tf --tiny

# video file with output saved (can save webcam like this too)
python detect_video.py --video path_to_file.mp4 --output ./detections/output.avi
```
Then you can find the detections in the `detections` folder.
<br>

### References
https://github.com/theAIGuysCode/Object-Detection-API

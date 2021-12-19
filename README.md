# Maskya

Mask Detector Application built using Tensorflow, Keras, and OpenCV.


## Background

This application is created due to help our campus get rid of maskless people.


## Methodology

1. Open up camera prompt via OpenCV.
2. Detect faces from camera video streams using **Multitask Cascaded Convolutional Networks (MTCNN)**.
3. Build the model using **Transfer Learning** technique by using **MobileNet V2** as the feature extraction layer, then add a custom fully connected layer to classify whether the input face is wearing a mask.
4. The mask detector model that we build before will be implemented in the real-time face detection system to detect whether all faces on the video streams are wearing a mask.


## Run Locally

Step-by-step to run this program.

#### Use the real-time mask detector

1. Clone or fork this repo.
2. Create a virtual environment.
3. Install all packages written inside ```requirements.txt```.
4. Run ```python main.py```

#### Train your own mask detector model

*Coming soon*

## Contributor

1. Jayaku Briliantio
2. Ferdy Nicolas
3. Jason Alexander
4. Martien Junaedi


## License

[MIT](./LICENSE)

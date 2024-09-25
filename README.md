Face Mask Detection System

This project utilizes a deep learning model to detect whether a person is wearing a face mask or not, using live video feed captured from a webcam. It is built using the MobileNetV2 architecture for image classification and OpenCV for real-time video processing.
Features

    Face Detection: Uses OpenCV's DNN module to detect faces in video frames.
    Mask Detection: Classifies each detected face as either "With Mask" or "Without Mask" using a trained MobileNetV2 model.
    Real-time Processing: Processes video stream from the webcam to detect and classify faces.

Requirements

    TensorFlow 2.x
    Keras
    OpenCV
    imutils
    Numpy
    scikit-learn
    matplotlib

Installation

    Clone the repository:

    bash

git clone https://github.com/eugenius-watchman/Face-Mask-Detection-Project-2.git

Install the required dependencies:

bash

    pip install -r requirements.txt

Training the Mask Detector Model

To train the face mask detector model, use the provided dataset containing two categories:

    with_mask: Images of people wearing face masks.
    without_mask: Images of people without face masks.

Steps:

    Load and preprocess the dataset.
    Train a MobileNetV2 model to classify images of faces as either with_mask or without_mask.
    Evaluate the model on the testing dataset and save the trained model.

Code:

python

# Train the mask detection model
model.fit(trainX, trainY, batch_size=BS, epochs=EPOCHS, validation_data=(testX, testY))

# Save the model for future use
model.save("mask_detector.model")

Running the Face Mask Detection System

    Load the pre-trained face detection and mask detection models:

    python

faceNet = cv2.dnn.readNet("face_detector/deploy.prototxt.txt", "face_detector/res10_300x300_ssd_iter_140000.caffemodel")
maskNet = load_model("mask_detector.model")

Start the video stream and perform real-time mask detection:

python

    vs = VideoStream(src=0).start()
    while True:
        frame = vs.read()
        (locs, preds) = detect_and_predict_mask(frame, faceNet, maskNet)
        ...

    Press q to quit the video stream.

Visualization

The project also plots the training loss and accuracy over time, using Matplotlib:

python

plt.plot(H.history["accuracy"], label="train_acc")
plt.plot(H.history["val_accuracy"], label="val_acc")
plt.savefig("plot.png")

Results

The system detects whether individuals are wearing masks or not, and marks the face regions with bounding boxes. Mask-wearing faces are marked with green boxes, while non-mask faces are marked with red boxes.

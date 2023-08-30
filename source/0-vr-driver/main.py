from keras.models import load_model  # TensorFlow is required for Keras to work
import cv2  # Install opencv-python
import numpy as np
import keyboard
import time

# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# Load the model
model = load_model("model/keras_Model.h5", compile=False)

# Load the labels
class_names = open("model/labels.txt", "r").readlines()

# CAMERA can be 0 or 1 based on default camera of your computer
camera = cv2.VideoCapture(0)

while True:
    # Grab the webcamera's image.
    ret, image = camera.read()

    image = cv2.flip(image, 1)  # 1 for horizontal flip

    # Resize the raw image into (224-height,224-width) pixels
    image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)

    # Show the image in a window
    cv2.imshow("Webcam Image", image)

    # Make the image a numpy array and reshape it to the models input shape.
    image = np.asarray(image, dtype=np.float32).reshape(1, 224, 224, 3)

    # Normalize the image array
    image = (image / 127.5) - 1

    # Predicts the model
    prediction = model.predict(image)

    

    index = np.argmax(prediction)
    class_name = class_names[index]
    confidence_score = prediction[0][index]

    if class_name[2:].strip() == "up":
         keyboard.press_and_release('up')
    
    elif class_name[2:].strip() == "down":
        keyboard.press_and_release('down')

    elif class_name[2:].strip() == "right":
        keyboard.press_and_release('right')

    elif class_name[2:].strip() == "left":
        keyboard.press_and_release('left')

    time.sleep(0.5)


    


    # Print prediction and confidence score
    #print("Class:", class_name)
    #print("Confidence Score:", str(np.round(confidence_score * 100))[:-2], "%")

    # Listen to the keyboard for presses.
    keyboard_input = cv2.waitKey(1)


    # 27 is the ASCII for the esc key on your keyboard.
    if keyboard_input == 27:
        break

camera.release()
cv2.destroyAllWindows()

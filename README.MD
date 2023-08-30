# Virtual Driving

## Repo Structure

```
    ├── docs                  # Contains project-related documents.
    ├── research              # Contains pre-research experiments.
    │   ├── 0-pre-prototype       # Contains pre-prototype (basic project).
    │   ├── 1-model.zip           # Contains trained model.
    ├── source                # Contains main source code.
    │   ├── 0-vr-driver           # Contains virtual driving app.
    │   ├── 1-tank-game           # Contains a web-based tank driving game.
    │   ├── 2-football-game       # Contains a web-based football game.
    └── README.MD             # Contains GitHub Readme Content.

```

## Introduction

The Virtual Driving Prototype is an innovative project that bridges the physical and virtual worlds using computer vision technology; a sheet of paper can be transformed into a joystick to control the game through a webcam, which helps to enhance the gaming experience, allowing players to control in-game vehicles with surprising precision.

The paper steering wheel becomes a tangible interface through the magic of computer vision and a standard webcam, bridging the gap between the player's physical actions and the virtual world on their screen.

It serves as a captivating example of how innovation can reshape the way we interact with and enjoy video games, offering a glimpse into the future of gaming experiences that are both entertaining and innovative.

**Note:** This AI model was trained with TechableMachine, a No-Code AI platform used for educational purposes. However, it is not suitable for production or enterprise purposes due to its lower performance quality.

## Technology Stack

- Python
- HTML
- CSS
- JavaScript
- Keras
- OpenCV
- Numpy
- Keyboard

## Setup

This repo's source folder has three sub-projects: Virtual Driving, Tank Game, and Football Game.

**Virtual Driving:** The Python prototype detects the car steering angles in the paper sheet through the webcam and triggers the arrow keys accordingly.

**Tank Game:** A simple web-based game written in HTML, CSS, and JavaScript to test the virtual driving prototype; the game's concept is to reach the end line without touching bombs (enemies). This game only uses arrow key inputs to negative.

**Football Game:** A simple web-based game written in HTML, CSS, and JavaScript to test the virtual driving prototype; the game's concept is to move the ball to the goalpost without touching opposing players.

**Note:** The Virtual Driving Prototype triggers the arrow keys (Up, Down, Left, Right) according to the webcam inputs; this prototype can be used for any purpose that needs to trigger arrow keys; it could also be applied to different games.

## Guide for AI Model Training (Techable Machine)

Teachable Machine is an innovative web-based platform developed by Google that empowers individuals, regardless of their coding experience, to delve into machine learning and artificial intelligence. This intuitive tool is a gateway for beginners and enthusiasts, allowing them to train their machine-learning models without extensive programming knowledge.

It is recognizing objects in images, identifying sounds, or classifying data. Teachable Machine simplifies the process by providing a user-friendly interface that enables users to teach their models through a simple demonstration process easily. This democratization of machine learning is transforming how people interact with AI, making it accessible and understandable to a broader audience and fostering creativity and innovation across various domains.

**Step 01:** Go to (https://teachablemachine.withgoogle.com)

**Step 02:** Click “Get Started Button”:

![IMAGE](docs/github-readme-content/1.jpg)

**Step 03:** Select “Image Project” and choose “Standard Image Model”:

![IMAGE](docs/github-readme-content/2.jpg)

**Step 04:** Collect Dataset

**Step 05:** Start Training

![IMAGE](docs/github-readme-content/3.jpg)

**Note:** After that, other steps are self-explanatory: create classes according to the needs; then add data to the relevant classes and click train to start the training process. Apart from this, advanced settings can be changed to improve performance and accuracy. The model can be uploaded online or downloaded locally; it also provides basic code to infer the model in Python, a Web App (TensorFlow | TensorFlowJS | TensorFlow Litte).

## Demonstration

![IMAGE](docs/github-readme-content/demo.gif)

## Social Links

- [Linkedin Profile](https://www.linkedin.com/in/gunarakulangunaretnam)
- [Facebook Profile](https://www.facebook.com/gunarakulangunaratnam)
- [Instagram Profile](https://www.instagram.com/gunarakulangunaretnam)
- [Twitter Profile ](https://twitter.com/gunarakulangr)
- [Kaggle Profile](https://www.kaggle.com/gunarakulangr)
- [TikTok Profile](https://www.tiktok.com/@gunarakulangunaretnam)
- [Youtube Profile](https://www.youtube.com/channel/UCMWkED5sabgVZSCKjZuRJXA)

# Behavioral Cloning for Autonomous Driving in Simulation

Behavorial Cloning allows a machine to imitate how a human drives a vehicle. The system 
learns an end-to-end deep learning model which takes each video frame as input and 
generates a control output. For the control output, only the steering angle is generated 
by the deep learning model and the speed is being maintained by a PD Controller. All these 
are done in a simulator. Below is the demo video of a vehicle running autonomously in the 
simulator. The vehicle can stay at the center of the lane for very challenging roads most of the time.

Please click on the image or the link to watch the demo video on YouTube.
[![Click to watch the demo video](https://img.youtube.com/vi/r1Xtko5LPMw/0.jpg)](http://www.youtube.com/watch?v=r1Xtko5LPMw)

Demo Video YoutTube link: https://youtu.be/r1Xtko5LPMw

The detailed explanation of the deep learning pipeline can be found in this notebook: https://github.com/paichul/Behavioral-Cloning/blob/master/demo.ipynb

# Key Deep Learning Concepts


# Software Usage Prerequisite
Make sure you have the right versions of the software installed: 
    - python==3.5.2
    - numpy
    - matplotlib
    - opencv3
    - scikit-learn
    - scikit-image
    - scipy
    - pandas
    - https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow_gpu-1.3.0-cp35-cp35m-linux_x86_64.whl
    - keras==2.0.7

# Usage Instructions

model.py uses Tensorflow and Keras to preprocess the images/control data (including data augmentation) and then generates a 
model called "model.h5".

To train the model, run:

"python model.py"

drive.py uses the model to generate steering angle prediction and uses a PD Controller to generate the speed command. Those commands are then sent to the simulator to drive the vehicle autonomously in the simulator. Please contact me at paichul@cs.stanford.edu for the simulator binary.

To run the trained model, type:

"drive.py model.h5"

Remember to start the simulator before running the model.

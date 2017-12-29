# Interpretable Behavioral Cloning in Simulation (Autonomous Driving Open Source)

Behavorial Cloning allows a machine to imitate how a human drives a vehicle. The system 
learns an end-to-end deep learning model which takes each video frame as input and 
generates a control output. Below is the demo video of a vehicle running autonomously in the 
simulator. The vehicle can stay at the center of the lane for very challenging roads most of the time.

[![Click to watch the full video](https://github.com/paichul/Behavioral-Cloning/blob/master/bc.gif)](https://www.youtube.com/watch?v=pNWlzoTTb_A)

# Features
- Convolutional Neural Network for Vehicle Steering Angle Control
- Residual Connection
- Batch Normalization
- Intermediate Layer Attention Map Visualization
- Unsupervised Road Segmentation
- Unsupervised Lane Marking Detection
- PI Controller for the Vehicle Speed Control

For the details please see the paper here.

# Software Dependencies
Make sure you have the right versions of the software installed: 
- python==3.5.2
- numpy
- matplotlib
- opencv3
- scikit-learn
- scipy
- pandas
- https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow_gpu-1.3.0-cp35-cp35m-linux_x86_64.whl
- keras==2.0.7

# Simulation Data Generation
Please contact me at paichul@cs.stanford.edu for the simulator binary.

# Training and Testing Instructions

model.py uses Tensorflow and Keras to preprocess the images/control data (including data augmentation) and then generates a 
model called "model.h5".

To train the model, run:

"python model.py"

drive.py uses the model to generate steering angle prediction and uses a PD Controller to generate the speed command. Those commands are then sent to the simulator to drive the vehicle autonomously in the simulator. 

To run the trained model, type:

"drive.py model.h5"

Remember to start the simulator before running the model.

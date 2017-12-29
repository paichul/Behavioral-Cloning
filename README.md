# Interpretable Behavioral Cloning in Simulation (Autonomous Driving Open Source)

Behavorial Cloning allows a machine to imitate how a human drives a vehicle. The system 
learns an end-to-end deep learning model which takes each video frame as input and 
generates a control output. Below is the demo video of a vehicle running autonomously in the 
simulator. The vehicle can stay at the center of the lane for very challenging roads most of the time.

[![Click to watch the full video](https://github.com/paichul/Behavioral-Cloning/blob/master/bc.gif)](https://www.youtube.com/watch?v=pNWlzoTTb_A)

# Features
- End-to-End Convolutional Neural Network for Vehicle Steering Angle Control
- Residual Connection
- Batch Normalization
- Intermediate Layer Attention Map Visualization
- Unsupervised Road Segmentation
- Unsupervised Lane Marking Detection
- PI Controller for Vehicle Speed Control

Please see the paper here for the detailed explanation of each feature.

# Intermediate Layer Attention Map Visualization
Unsupervised Road Segmentation: According to the Attention Map Visualization, we see that 
Convolutional Layer 1 has learned to perform Road Segmentation even without 
any semantic scene segmentation label information, but only the steering angle control supervision.
The green region shows irrelevant image pixels for the steering angle prediction and the red region (the road)
shows the relevant image pixels that the model pay attention to for the steering angle prediction.

![](https://github.com/paichul/Behavioral-Cloning/blob/master/unsupervised%20road%20segmentation.png)

Unsupervised Lane Marking Detection: According to the Attention Map Visualization, we see that 
Convolutional Layer 2 and Convoluational Layer 3 have learned to perform Lane Marking Detection even without 
any lane marking label information, but only the steering angle control supervision.
The green/blue regions show less relevant image pixels for the steering angle prediction and the red region (the lane marking) shows the more relevant image pixels that the model pay attention to for the steering angle prediction.

![](https://github.com/paichul/Behavioral-Cloning/blob/master/unsupervised%20lane%20marking%20detection.png)


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
Please see the paper here for the detailed explanation of the simulation data generation procedure.
Please send an email to paichul@cs.stanford.edu for the simulator binary.

# Training and Testing Instructions
- model.py: model training pipeline from data preprocessing, data augmentation, model specification to optimization. The training outputs a model called "model.h5" that can be used by the simulator to run the vehicle autonomously.
- drive.py: model testing pipeline including a PI Controller to generate the speed command and it lods the model.h5 to generate the steering angle prediction. It also sends the speed and steering angle command to the simulator to maneuver the vehicle autonomously in the simulator.
- visualize.py: visualization pipeline for interpretability.
- tools.py: data preprocessing utility tools

To train the model, run:

"python model.py"


To run the trained model, type:

"python drive.py model.h5"

Remember to start the simulator before running the model.

To visualize different layer of the model, run:

"python visualize.py [layer_name]"

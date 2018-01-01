# Interpretable Behavioral Cloning in Simulation (Autonomous Driving Open Source)

Behavorial Cloning allows a machine to imitate how a human drives a vehicle. The system 
learns an end-to-end deep learning model which takes each video frame as input and 
generates a control output. Below is the demo video of a vehicle running autonomously in the 
simulator. The vehicle can stay at the center of the lane for very challenging roads most of the time.

[![Click to watch the full video](https://github.com/paichul/Behavioral-Cloning/blob/master/bc.gif)](https://www.youtube.com/watch?v=pNWlzoTTb_A)

See the 6 minute full video here: https://www.youtube.com/watch?v=pNWlzoTTb_A

# Features
- End-to-End Convolutional Neural Network for Vehicle Steering Angle Control
- Shortcut Connection
- Batch Normalization
- Intermediate Layer Activation (Attention) Map Visualization
- Unsupervised Road Segmentation
- Unsupervised Lane Marking Detection
- PI Controller for Vehicle Speed Control

Please see [the paper](https://goo.gl/1xiLWf) here for the detailed explanation of each feature.



# Layer Activation (Attention) Map Visualization
<H2>Unsupervised Road Segmentation</H2>
According to the Activation (Attention) Map Visualization for the first Convolutional layer, we see that Convolutional Layer 1 (Conv1) has learned to perform Road Segmentation even without 
any semantic scene segmentation label information, but only the steering angle control supervision.
The green region shows image pixels the model pays relatively less attention to for the steering angle prediction and the red region (the road) shows the image pixels that the model pays substantially more attention to for the steering angle prediction.

![](https://github.com/paichul/Behavioral-Cloning/blob/master/images/unsupervised%20road%20segmentation.png)

<H2>Unsupervised Lane Marking Detection</H2>
According to the Activation (Attention) Map Visualization for the second and third convolutional layers, we see that 
Convolutional Layers 2 and 3 (Conv2 and Conv3) have learned to perform Lane Marking Detection even without 
any lane marking label information, but only the steering angle control supervision.
The green/blue regions show image pixels that the model pays relatively less attention to for the steering angle prediction and the red region (the lane marking) shows the image pixels that the model pays substantially more attention to for the steering angle prediction.

![](https://github.com/paichul/Behavioral-Cloning/blob/master/images/unsupervised%20lane%20marking%20detection.png)

<H2>Information Bottleneck and Individual Neuron Activation Visualization </H2>
The paper further describes how the fourth convolutional layer makes high-level decision based on the low-level features learned by the previous convolutional layers (Conv1, Conv2, Conv3 layers) and describes the interaction between Layer Activation (Attention) Map visualization and Individual Neuron Activation visualization. As well, the paper describes how the fifth convolutional layer makes high-level decisions by compressing the features as much as possible while achieving as high steering angle prediction accuracy as possible; and describes the potential interaction between Information Bottleneck and Layer Activation (Attention) Map visualization. See Section 3.5 of [the paper](https://goo.gl/1xiLWf) for the detailed explanation.

# Software Dependencies
Make sure you have the right versions of the software installed: 
- python 3.5.2
- numpy
- matplotlib
- opencv3
- scikit-learn
- scipy
- pandas
- tensorflow gpu-1.3.0 https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow_gpu-1.3.0-cp35-cp35m-linux_x86_64.whl
- keras 2.0.7

# Simulation Data Generation
Please see Section 3.1 of [the paper](https://goo.gl/1xiLWf) for the detailed explanation of how to generate the simulation data for training/testing the model. Please send an email to paichul@cs.stanford.edu for the simulator binary.

# Training and Testing Instructions
- model.py: model training pipeline from data preprocessing, data augmentation, model specification to training optimization. The training process generates a model called "model.h5" that can be used by the simulator to run the vehicle autonomously.
- drive.py: model testing pipeline including a PI Controller to generate the speed command and it loads the model "model.h5" to generate the steering angle prediction. It also sends the speed and steering angle commands to the simulator to maneuver the vehicle autonomously in the simulator.
- visualize.py: visualization pipeline using Layer Activation (Attention) Map Visualization for model interpretability.
- tools.py: data preprocessing utility tools

To train the model, run:

"python model.py"

To run the trained model, type:

"python drive.py model.h5"

Remember to start the simulator before running the model.

To visualize different layer of the model, run:

"python visualize.py [layer_name]"

# Publication
See the paper ["Interpretable Behavioral Cloning in Simulation"](https://goo.gl/1xiLWf) for this open source project.

# Simulation to Reality Transfer
This open source project is a small component of the larger Simulation-To-Reality Transfer for Autonomous Driving/Robots framework. The rest of the framework will be open sourced soon. Potential future open source features include:
- Generative Models (such as Adverserial Generative Networks)
- Domain Adoptation
- Transfer Learning
- Multi-Task Learning
- Meta Learning
- Adverserial Example

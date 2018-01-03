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
Activation Map allows people to visualize how the different neural network layer pays attention to different parts of the image with respect to a discrete class label. This thus allows us to reason about inference attribution.

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

<H2>High-level Convolutional Layer Shortcut Connection</H2>
Adding shortcut connection from Conv5 to concatenate with the Dense4 improves the steering angle prediction accuracy because Conv5 is making high-level decisions just like Dense4, so merging Conv5 and Dense4 together works well. The shortcut connection allows the gradient to flow directly to Conv5 to make it learn this high-level decision making more effectively. As well, the shortcut connection can somewhat enables the model to decide if it wants to depend on the Dense4 decision or Conv5 decision, or both. In other words, it allows multiple decision makers to cast the votes,

<H2>Low-level Convolutional Layer Batch Normalization</H2>
Batch Normalization is added right after the first convolutional layer (Conv1) and the second convolutional layer (Conv2) and before their activation functions. Adding Batch Normalization at lower convolutional layers allows the model to be less sensitive to lower-level image features/input distribution and thus can generalize better.

<H2> Conv5 Visualization and Information Bottleneck (under Investigation) </H2>
The image below shows it pays attention ahead (dark red region) for steering the vehicle straight ahead. One interesting observation is that the image is mostly red but one can still spot the darker red region to find the model’s attention focus. The activation (attention) map visualization is mostly red-ish possibly because at the highest-level, convolutional layer Conv5 tries to compress away as much irrelevant information as possible to make sure the high-level features are highly relevant to the steering angle prediction. This compression phenomenon has also been observed in related Information Bottleneck research work by S. Keerthi: "On The Information Bottleneck" in ICLR, 2018. However, the interaction between the activation map layer visualization and Information Bottleneck (compression and accuracy tradeoff) is still under active investigation.

<H2> Gradient Ascent Based Neuron Visualization (under Investigation)</H2>


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

# References


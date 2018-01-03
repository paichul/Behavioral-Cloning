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

Please see [the paper](https://goo.gl/KTzoLT) here for the detailed explanation of each feature.


# Layer Activation (Attention) Map Visualization
Activation Map allows people to visualize how the different neural network layer pays attention to different parts of the image with respect to the control output (steering angle). The Activation Map Visualization approach in this research work allows us to see how a neural network performs holistic scene understanding for the steering angle control prediction.

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
Adding shortcut connection from Conv5 (Fifth Convolutional Layer) to concatenate with the Dense4 (Fourth Fully Connected Layer) improves the steering angle prediction accuracy because Conv5 is making high-level decisions just like Dense4, so merging Conv5 and Dense4 together works well. The shortcut connection allows the gradient to flow directly to Conv5 to make it learn this high-level decision making more effectively. As well, the shortcut connection can somewhat enables the model to decide if it wants to depend on the Dense4 decision or Conv5 decision, or both. In other words, it allows multiple decision makers to cast the votes,

<H2>Low-level Convolutional Layer Batch Normalization</H2>
Batch Normalization is added right after the first convolutional layer (Conv1) and the second convolutional layer (Conv2) and before their activation functions. Adding Batch Normalization at lower convolutional layers allows the model to be less sensitive to lower-level image features/input distribution and thus can generalize better.

<H2> Conv4 Visualization for High-Level Decision Making</H2>
The image below shows how the fourth convolutional layer (Conv4) pays attention ahead and the lane marking (red regions) in terms of the abstract representation learned by the lower convolutional layers for steering the vehicle straight ahead. 

![](https://github.com/paichul/Behavioral-Cloning/blob/master/images/high-level%20decision%20making.png)

<H2> Gradient Ascent Based Neuron Activation Visualization </H2>
We can learn an input image that maximally activates a certain neuron in the neural network model using Gradient Ascent Based Activation approach as summarized in https://distill.pub/2017/feature-visualization/. The interaction between layer activation map visualization (collective multi-neuron interactions) and gradient ascent based neuron activation visualization is currently work in progress.

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
See the paper ["Interpretable Behavioral Cloning in Simulation"](https://goo.gl/KTzoLT) for this open source project.

# Simulation to Reality Transfer
This open source project is a small component of the larger Simulation-To-Reality Transfer for Autonomous Driving/Robots framework. The rest of the framework will be open sourced soon. Potential future open source features include:
- Generative Models (such as Adverserial Generative Networks)
- Domain Adoptation
- Transfer Learning
- Multi-Task Learning
- Meta Learning
- Adverserial Example

# References
- Volodymyr Mnih, "Playing Atari with Deep Reinforcement Learning," 2013.
- P. Lin, "Behavioral Cloning for Autonomous Driving in Simulation," https://youtu.be/pNWlzoTTb_A.
- Ian J. Goodfellow, "Generative Adversarial Networks," in NIPS, 2014.
- Bolei Zhou, "Learning Deep Features for Discriminative Localization," in CVPR, 2016.
- Ramprasaath R. Selvaraju, "Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization," in ICCV, 2017.
- Jost Tobias
- Springenberg, "Striving for Simplicity: The All Convolutional Net," in ICLR, 2015.
- Anh Nguyen, "Multifaceted Feature Visualization: Uncovering the Different Types of Features Learned By Each Neuron in Deep Neural Networks," in ICML, 2016.
- N. T. Ravid Schwartz-Ziv, "Opening the black box of Deep Neural Networks," in ICRI-CI, 2017.
- J. B. Diederik P. Kingma, "Adam: A Method for Stochastic Optimization," in ICRL, 2015.
- S. Keerthi, "On The Information Bottleneck,"inICLR,2018.
- D. A. Pomerleau, "ALVINN, an autonomous land vehicle in a neural network.," Technical report, Carnegie Mellon University, 1989.
- Mariusz Bojarski, "End to End Learning for Self-Driving Cars," in CVPR, 2016.
- Jun-Yan Zhu, "Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial"
- T. B. J. K. Ming-Yu Liu, "Unsupervised Image-to-ImageTranslationNetworks,"in NIPS, 2017.
- Konstantinos Bousmalis, "Using Simulation and Domain Adaptation to Improve Efficiency of Deep Robotic Grasping," in CVPR, 2017.

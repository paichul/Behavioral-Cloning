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

The detailed explanation of the deep learning pipeline source code can be found in this notebook: https://github.com/paichul/Behavioral-Cloning/blob/master/demo.ipynb

# Key Deep Learning Insights
- Larger filter sizes for the first few convolutional layers give better performance because larger filter size alllows it to capture more spatial information.
- It's also important to decrease the filter size as the feature map resolution decreases after several convolutional layers becuase decreasing the feature map size forces the model to learn more relevant high-level image features for the steering prediction.
- I actively monitored the training accuracy and validation accuracy for different models I tried. The accuracy is based on Mean Sqaured Error between the ground truth steering angle and the predicted steering angle. I found that using fewer than 5 convolutional layers would cause the model to overfit as the validation accuracy is substantially lower than the training accuray. By doing an error analysis, I found that that this model cannot deal with large turns (that is, to generate big steering angle command). Having more than 5 convolutional layers does not improve the performance for this simulation data. Thus, 5 convolutional layers works best in this demo.
- To avoid overfitting, I applied dropout with probablity of 0.5 right after th last convolutional layer. Adding dropout elsewhere did not improve the performance. 

# Interpretable Deep Learning Insights
- I extended Grad-CAM (Gradient Class Acitvation Map) to allow it to work with continouous control output (steering angle). Grad-CAM was designed to allow people to see which part of the input image the convolutional neural network puts more weight on based on the current discrete class label prediction. This visualization can be done for each layer to see how each layer pays attention to different parts of the input image. However, for autonomous driving, the steering angle command is continuous, not discrete. Thus, I modified the original algorithm to make it work with continuous control. Below is an example of the visualization image done for the first convoluational layer. Red color means higher attention weight and blue color means lower attention weight.




- I found that lower-level convolutional layers shows more precise details on which parts of images those layers pay attention to; whereas, the higher-level convolutional layers tend to generate less precise attention details but more high-level semantics such as putting weight on the left or on the right part of the image.

# Key Computer Vision Insights
-

# Key Robotics Control Insights
- 

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

# Simulation Data Generation
Please contact me at paichul@cs.stanford.edu for the simulator binary.

# Usage Instructions

model.py uses Tensorflow and Keras to preprocess the images/control data (including data augmentation) and then generates a 
model called "model.h5".

To train the model, run:

"python model.py"

drive.py uses the model to generate steering angle prediction and uses a PD Controller to generate the speed command. Those commands are then sent to the simulator to drive the vehicle autonomously in the simulator. 

To run the trained model, type:

"drive.py model.h5"

Remember to start the simulator before running the model.

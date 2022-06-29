# **Behavioral Cloning** 

## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[arch1]: ./doc/lenet_arch.jpeg
[arch2]: ./doc/nvidia_arch.png
[set1c]: ./doc/center.jpg
[set1l]: ./doc/left.jpg
[set1r]: ./doc/right.jpg
[set2c]: ./doc/center2.jpg
[set2l]: ./doc/left2.jpg
[set2r]: ./doc/right2.jpg
[set3c]: ./doc/center3.jpg
[set3l]: ./doc/left3.jpg
[set3r]: ./doc/right3.jpg
[frame1]: ./doc/videoframe1.jpg
[frame2]: ./doc/videoframe2.jpg
[frame3]: ./doc/videoframe3.jpg

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```
before launching the simulator and entering autonomous mode.

#### 3. Submission code is usable and readable

The clone.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My I trained the neural network with two architectures, LeNet and Nvidia, and implemented them as described in the instructions for this project. 

The model includes RELU layers to introduce nonlinearity (see the methods in clone.py to create the LeNet model [lines 48-58], and the Nvidia model [lines 60-72]), and the data is normalized in the model using a Keras lambda layer (clone.py line 44). 

#### 2. Attempts to reduce overfitting in the model

Overfitting was prevented by observing the validation loss compared to the overall loss. At times  when the validation loss was continuously increasing, steps were taken to prevent overfitting by either reducing the number of epochs or by changing the preprocessing steps in the networks (one one occasion).

The training and validation sets were shuffled and randomised to ensure that the model was not overfitting (clone.py line 76). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (clone.py line 75).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, driving the track in both directions, and augmenting the images with flipping to achieve this. 

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to make a simple neural network capable of procucing basic input/ouput and staying on the track for some distance, then improving it gradually by updating the architecture or adjusting the data.

My first step was to use a simple neural network model with a single flattened layer for the image connected to one output node. I took the idea for this from the project instructions. The vehicle was able to follow the road, but it was slow and stayed on the right edge of the road.

Once I knew that the basic neural net was on the right track, I swapped the rudimentary architecture for the LeNet architecture, and the model did much better, but still had some trouble correcting its position if it drifted too far to the edge of the road.

This is a visualisation of the LeNet architecture:

![LeNet Architecture][arch1]
    
At this stage, the car drove well and almost all the way around the track. But after it crossed the bridge, it got confused by the dirt track and drove on that for a while before eventually going into a rock and stopping.

The final improvement was to implement the Nvidia architecture, which allowed the vehicle to finish a lap of the road. The car drove smoothly (if slowly) around the road and stayed within the yellow lines for the duration.

#### 2. Final Model Architecture

The final model architecture (clone.py lines 60-72) was modeled after the Nvidia architecture: 5 convolutional layers followed by 4 fully-connected layers.

The network can be visualised like this:

![Nvidia Architecture][arch2]

#### 3. Creation of the Training Set & Training Process

From the beginning of the project, I chose to use good training data by practicing my drives on the track until I was confident my driving was smooth, I was driving in the centre of the road, and I wasn't recording any obvious mistakes. This took some time, mainly due to mouse sensitivity in the virtual machine and no options for adjusting it in the settings of the simulator. After I recorded a good lap going in one direction, I then turned the vehicle around and recorded another going the opposite direction. This is the only training data I recorded, and was sufficient to complete the project.

To augment the data sat, I also flipped images and angles to help generalise the model.

I also used images from the right and left cameras, and added a correction to the steering values to help direct the car back to the centre of the road. Images from all three cameras at the same time look like this:

![Left Camera][set1l]
![Centre Camera][set1c]
![Right Camera][set1r]

![Left Camera][set2l]
![Centre Camera][set2c]
![Right Camera][set2r]

![Left Camera][set3l]
![Centre Camera][set3c]
![Right Camera][set3r]

I had two preprocessing steps, both impelemented by lambda layers in Keras. The steps were:
1. Normalising and mean-centreing the data (clone.py line 44)
2. Cropping the images to remove the terrain/sky and the hood of the car (clone.py line 45)

In the instructions, David cropped the images slightly differently by removing the top 70 instead. However, there was still a little road data in those pixels that I wanted to use, so I chose 60 instead.

I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The number of epochs in the end was 5. Earlier in the development process I was using 20, but as the model became more complex and had more images to train from, 20 became impractical. I used 5 because I could still see that the validation loss was still able to drop or balance out at that stage.

I used an adam optimizer so that manually training the learning rate wasn't necessary.

#### Video

After completing a lap, I made a video containing the centre-camera footage of the run. It can be found in [track1.mp4](./track1.mp4).

Some frames from the video:

![Sample Frame 1][frame1]
![Sample Frame 2][frame2]
![Sample Frame 3][frame3]


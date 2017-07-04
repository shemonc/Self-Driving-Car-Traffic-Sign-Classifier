#**Traffic Sign Recognition** 

##Writeup Template

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/visualization.png "Visualization"
[image2]: ./examples/grayscale.png "Grayscaling"
[image3]: ./examples/random_noise.png "Random Noise"
[image4]: ./examples/100km.png "Traffic Sign 100k/h"
[image5]: ./examples/yield.png "Traffic Sign Yield"
[image6]: ./examples/stop.png "Traffic Sign Stop"
[image7]: ./examples/slippery_road.png "Traffic Sign Slippery road"
[image8]: ./examples/ahead_only.png "Traffic Sign Ahead Only"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/shemonc/Self-Driving-Car-Traffic-Sign-Classifier/blob/master/Traffic_Sign_Classifier.ipynb)

###Data Set Summary & Exploration

####1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the numpy library to calculate summary statistics of the traffic
signs data set:


* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is  (32, 32, 3)
* The number of unique classes/labels in the data set is 43

####2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the data ...

![alt text][image1]

###Design and Test a Model Architecture

####1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided to convert the images to grayscale because ...

My dataset preprocessing consisted of:
Converting to grayscale - I use gray scale image in LeNet architecture using TensorFlow. Training time is less, this is helpful while training this on my macbook

As a last step, I normalized the image data because,

Normalizing the data to the range (-1,1) , This was done using the equation X_train_normalized = (X_train - mean_of_X_train)/standard_deviation_of_X_train, bellow shows mean and variance 
of my Train, Valid and Test data before and after normalization. After normalization they mean very close to zero and variance are equal to each other.

== Data before Normalization ==
Train Data  82.677589037
Valid Data  83.5564273756
Test Data  82.1484603612
Train Variance [0-127]  4054.4861014
Train Variance [128-255]  4655.41963214
Train Variance [256-384]  4591.37094437

== Data after Normalization ==
Train Data  -1.31869359324e-19
Valid Data  -2.349678359e-19
Test Data  -4.02012665554e-18
Train Variance [0-127]  0.880446952967
Train Variance [128-255]  1.00922009169
Train Variance [256-384]  0.992372176702

As seen above data are very close zero mean and equal variance which is a good starting point for optimizing the loss to avoid too big or too small. A badly condition problem means that the optimizer has to do a lot of searching to go and find a good solution.


![alt text][image2]

I decided to generate additional data because,

Some of the classes were represented far more than others sometime 10 fold higher. This is a problem because the lack of balance in the class data will lead into become biased toward the classes with more data points. Instead of omitting valid data from the classes with more datapoints I generate more data for the classes those are less than 500 by using a Gaussian bluring of the real image with a kernel size of 3, 5 etc.

Here is an example of an original image and an augmented image:

![alt text][image3]

The difference between the original data set and the augmented data set is the following ... 
in my cases the augmented dataset are the blurred images with the original images, which satisfy 2 purpose 
i) with balanced data for all classes, there are less chance of overfitting
ii) Blurring reduce noise and is good for image processing. I kind of randomize the kernel function to do the blurring a bit differently for the same image to produce more than 1 image.

####2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 Gray image   							| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6  				|
| Convolution 5x5	    | 1x1 stride, valid padding, output 10x10x16    |
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x16  				    |
| Flatten layer			| input 5x5x16, output 400        				|
| Fully connected		| input 400, output 120        					|
| RELU					|												|
| Dropout				|												|
| Fully connected		| input 120, output 84    						|
| RELU					|												|
| Dropout				|												|
| Fully connected		| input 84, output 43    						|
| Activation(Softmax)	| 43        									|
|						|												|
 


####3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used

EPOCHS of 14
BATCH_SIZE of 128
Learning rate of rate = 0.001

* I selected a lower learning rate as it keeps on going and better versus a higher learning rate which learns faster but also plateaus earlier. low learning rate also required
  higher number of Epochs 
 
* During training target is to minimize the training loss which is a function of weights and biases. Initializing the weights with random numbers from a normal distribution is good 
  practice and is done here. Randomizing the weights helps the model from becoming stuck in the same place every time it is trained. Similarly, choosing weights from a normal 
  distribution prevents any one weight from overwhelming other weights. following tensorflow truncated_normal() function was used to select the weight.
  mu = 0
  sigma = 0.1
  conv1_W = tf.Variable(tf.truncated_normal(shape=(5, 5, 1, 6), mean = mu, stddev = sigma))
  for numerical stability and better optimization training/validating/testing data was normalized. Also I picked a low sigma value above which means small peak, less opinionistic that
  can train better
* Also use Adam Optimizer over SGD.
* Use Dropout (50%) for regularization to force the network to lean and avoid overfitting


####4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* validation set accuracy was 94%
* test set accuracy was 91.5%

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
  - Original LeNet Model Architecture was tried out here wtih 2 dropout added at fully connected layer 3 and 4 to avoid overfitting
* What were some problems with the initial architecture?

* How was the architecture adjusted and why was it adjusted?
  - As explained above 2 dropout layer were added to compensate for overfitting

* Which parameters were tuned? How were they adjusted and why?
   - please see section #2 and #3 above

* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?

If a well known architecture was chosen:
* What architecture was chosen?
  I use the Original LeNet Model Architecture¶
* Why did you believe it would be relevant to the traffic sign application?
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
 

###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

The first image might be difficult to classify because ...

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 100 km/h	      		| 100 km/h 					 					|	
| Yield					| Yield											|
| Stop Sign      		| Stop sign   									| 
| Slippery Road			| Slippery Road      							|
| Ahead Only			| Ahead Only    								|
| Keep left				| Keep left     								|

I take the new images from official GTSRB training set, which are not in 32x32x3 size, I resize them to 32x32x3 inorder to feed into Yan LeCun's LeNet model.
The model was able to correctly guess 6 of the 6 traffic signs in above table, which gives an accuracy of 100%, which is better than my validation accuracy which is 94%

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a stop sign (probability of 0.6), and the image does contain a stop sign. The top six soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1	      				| 100 km/h 					 					|	
| 1						| Yield											|
| 1         			| Stop sign   									| 
| 1				    	| Slippery Road      							|
| 1						| Ahead Only    								|
| 1						| Keep left     								|

The model identifies the above signs with 100% accuracy - better than the 94% validation accuracy and the 91.5% test accuracy. The model works well in new data which is good but in real world
with not so clear image like those given in this experiment the model accuracy might not be as good as in this lab.


### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
####1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?

Feature maps are show for class_id 38 (keep right) with 6 images.
In LeNet-5 model, Units in a Feature map are all constrained to perform the same operation on different parts of the image. As is shown in this experiment, the first layer of LeNet-5 are organized in
6 planes, each of which is a feature map. A unit in a feature map has 25 inputs connected to a 5x5 area in the input, called the receptive field of the unit. All the units in a feature map share the same
set of 25 weights and the same bias so they detect the same feature at all possible locations on the input.

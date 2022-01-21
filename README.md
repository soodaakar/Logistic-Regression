# Overview
The goal of the assignment is to differentiate whether a patient has diabetes
(class 1) or not (class 0). In the first part of the assignment we are asked to
build a logistic regression model to di↵erentiate between the two classes.

## Dataset
The dataset was extracted from Pima Indians Diabetes Database. The dataset
consists of female patients aged above 21 years. It consists of 768 instances and
8 features. The dataset is divided into three parts: train, validation and test.
The train dataset has 460 instances, validation has 154 instances and test also
has 154 instances.

## Python Implementation
I have used Jupyter Notebook IDE for implementation of the project and the
same has been shared.

### Data Preprocessing
The dataset was read using pandas dataframe. The dataset was checked for any
missing values, the percentage distribution of the target variable and statistics
of each independent variable. Certain independent variable in the dataset were
standardized using the formula below:

<img src="https://bit.ly/3AjXWcQ" align="center" border="0" alt="z = (x_i - u)/ \sigma" width="162" height="29" />

Further the dataset was divided into 3 parts: Train, Validation and Test. I used
stratified splitting and the split size was 60-20-20 (train-validation-test).

## LogisticRegression
Logistic regression was coded in python using the below set of equations:
Below equation multiples the weight with each features and add bias to it

<img src="http://www.sciweavers.org/tex2img.php?eq=z%3Dw%5ET.X%2Bb&bc=White&fc=Black&im=jpg&fs=18&ff=modern&edit=0" align="center" border="0" alt="z=w^T.X+b" width="154" height="27" />

After calculating the above step we pass it on to the sigmoid function

<img src="http://www.sciweavers.org/tex2img.php?eq=f%28z%29%20%3D%201%2F%201%20%2B%20e%5Ez%20%3De%5Ez%2F1%20%2B%20e%5Ez&bc=White&fc=Black&im=jpg&fs=18&ff=modern&edit=0" align="center" border="0" alt="f(z) = 1/ 1 + e^z =e^z/1 + e^z" width="302" height="29" />

Once the value is returned from the sigmoid function we check for the loss using
the equation

<img src="http://www.sciweavers.org/tex2img.php?eq=j%28%5Ctheta%29%20%3D%20-%20%5Cfrac%7B1%7D%7Bn%7D%20%5Csum%5Climits_%7Bi%3D1%7D%5En%20%5By_i%20%2A%20log_e%28%5Chat%7By_i%7D%29%20%2B%20%281-y_i%29%20%2A%20log_e%281-%5Chat%7By_i%7D%29%20&bc=White&fc=Black&im=jpg&fs=18&ff=modern&edit=0" align="center" border="0" alt="j(\theta) = - \frac{1}{n} \sum\limits_{i=1}^n [y_i * log_e(\hat{y_i}) + (1-y_i) * log_e(1-\hat{y_i}) " width="542" height="71" />

On the basis of loss we update our weights and bias using:

<img src="http://www.sciweavers.org/tex2img.php?eq=w%20%3D%20w%20-%20%5Calpha%20%2A%20dw&bc=White&fc=Black&im=jpg&fs=18&ff=modern&edit=0" align="center" border="0" alt="w = w - \alpha * dw" width="175" height="23" />

<img src="http://www.sciweavers.org/tex2img.php?eq=b%20%3D%20b%20-%20%5Calpha%20%2A%20db&bc=White&fc=Black&im=jpg&fs=18&ff=modern&edit=0" align="center" border="0" alt="b = b - \alpha * db" width="154" height="23" />

### Results

Two logistic regression models were created having learning rate as 0.05 (Model
A) and 0.01 (Model B). Both the model ran for the same number of iterations
which was 1000. Below are loss curves for the same.

![image](https://user-images.githubusercontent.com/47882482/150602929-cd685b49-581a-4c24-87d8-d3061c4f0a87.png)

Figure 1: This is an image showing the classification report when learning rate
is 0.05.

![image](https://user-images.githubusercontent.com/47882482/150610095-ef8b73a7-80a1-482d-9c35-4583121c8625.png)

Figure 2: This is an image showing the classification report when learning rate
is 0.01.

From the image 2 it can be seen that model with learning rate 0.01 hasn’t
converged and can be trained on more epochs.
Model A achieved an accuracy of 78% on the train set and 77% on the validation
set whereas Model B achieved an accuracy of 78% on train and 76% on the
validation set.

Below are the classification report of the above two models:

![image](https://user-images.githubusercontent.com/47882482/150610153-b69752e1-1054-4990-a8bf-067627916be1.png)

Figure 3: This is an image shows the loss curve when learning rate is 0.05.

![image](https://user-images.githubusercontent.com/47882482/150610199-af8e51be-eff4-4dca-854e-88b1a19b6f43.png)

Figure 4: This is an image shows the loss curve when learning rate is 0.01.
With the both the models performing almost the same, I have chosen model
with learning rate 0.05 as it has better accuracy on the validation set.

### Test Result
On the test data the model achieved an accuracy of 79%. Below is the classifi-
cation report on the test data:

![image](https://user-images.githubusercontent.com/47882482/150610255-dd4ce070-a5fc-436d-a4cc-13826d5972cb.png)

Figure 5: This is an image shows the loss curve when learning rate is 0.01.

## Neural Networks
Neural network is a set of system which was inspired by biological neural net-
work also know as brain. They are a collection of neurons, hidden layers and
activation function. Inside a neural network there are two type of passes known
as forward pass or forward propagation and backward pass or backward propa-
gation.

### Forward Propagation
In forward propagation we assign weights to each and every connection between
neurons. Each layer has it’s own set of activation function which can vary
based on each and every hidden layer. Below are the steps involved in forward
propagation:

<img src="http://www.sciweavers.org/tex2img.php?eq=z%3Dw%5ET.X%2Bb&bc=White&fc=Black&im=jpg&fs=18&ff=modern&edit=0" align="center" border="0" alt="z=w^T.X+b" width="154" height="27" />

The above equation will assign weights to each neuron connection and one bias
per layer. Once this is done we can apply di↵erent type of activation function
like Sigmoid or Relu.
Sigmoid activation function equation:

<img src="http://www.sciweavers.org/tex2img.php?eq=f%28z%29%20%3D%201%2F%201%20%2B%20e%5Ez%20%3De%5Ez%2F1%20%2B%20e%5Ez&bc=White&fc=Black&im=jpg&fs=18&ff=modern&edit=0" align="center" border="0" alt="f(z) = 1/ 1 + e^z =e^z/1 + e^z" width="302" height="29" />

![image](https://user-images.githubusercontent.com/47882482/150610295-d9fb9695-c8dd-4bd6-867c-d93168fb3e86.png)

Figure 6: This is an image showing sigmoid activation function.
Relu activation function equation:

![](http://www.sciweavers.org/tex2img.php?eq=%5Ctext%7By%7D%20%3D%20%20%20%20%5Cbegin%7Bcases%7D%20%20%20%20%5Cmbox%7B%241%24%7D%20%26%20%5Cmbox%7Bif%20%7D%20x%20%3E%200%5C%5C%20%20%20%20%5Cmbox%7B%240%24%7D%20%26%20%5Cmbox%7Bif%20%7D%20x%20%5Cleq%200%20%20%20%20%5Cend%7Bcases%7D&bc=White&fc=Black&im=jpg&fs=18&ff=modern&edit=0)

![image](https://user-images.githubusercontent.com/47882482/150610323-145c4997-c08d-4e76-8810-fa90639ec045.png)

Figure 7: This is an image showing relu activation function.
There are many more activation function which could be used like tanh, leaky
relu, etc but the above two are ones which are used in my model building process.

### Backward Propagation
Once we have completed the forward propagation we need to compute the loss
and try to minimize the loss by updating weights and bias for each and every
neuron connection, exactly what is done in backward propagation. Loss was
calculated in the last layer having just one neuron by using binary cross-entropy.
Below is the formula:

<img src="http://www.sciweavers.org/tex2img.php?eq=j%28%5Ctheta%29%20%3D%20-%20%5Cfrac%7B1%7D%7Bn%7D%20%5Csum%5Climits_%7Bi%3D1%7D%5En%20%5By_i%20%2A%20log_e%28%5Chat%7By_i%7D%29%20%2B%20%281-y_i%29%20%2A%20log_e%281-%5Chat%7By_i%7D%29%20&bc=White&fc=Black&im=jpg&fs=18&ff=modern&edit=0" align="center" border="0" alt="j(\theta) = - \frac{1}{n} \sum\limits_{i=1}^n [y_i * log_e(\hat{y_i}) + (1-y_i) * log_e(1-\hat{y_i}) " width="542" height="71" />

On the basis of loss we update our weights for each and every neuron connection
and bias for each hidden layer using:

<img src="http://www.sciweavers.org/tex2img.php?eq=w%20%3D%20w%20-%20%5Calpha%20%2A%20dw&bc=White&fc=Black&im=jpg&fs=18&ff=modern&edit=0" align="center" border="0" alt="w = w - \alpha * dw" width="175" height="23" />

<img src="http://www.sciweavers.org/tex2img.php?eq=b%20%3D%20b%20-%20%5Calpha%20%2A%20db&bc=White&fc=Black&im=jpg&fs=18&ff=modern&edit=0" align="center" border="0" alt="b = b - \alpha * db" width="154" height="23" />

The above whole process comes under backward propagation. A complete cycle
of forward propagation and backward propagation is known has one epoch.

### First Model
I have trained my first model by taking only one hidden layer and inside that
hidden layer took 32 neurons. For all the models build a standard learning rate
of 0.001 and batch size of 64 was taken. Below are the training and validations
loss and accuracy curves:


(a) Loss curves for train and validation | (b) Accuracy curves for train and validation
:-------------------------:|:-------------------------:
![image](https://user-images.githubusercontent.com/47882482/150610394-76e5f7c9-6343-4a20-a23b-2b9b29d2a3b7.png) | ![image](https://user-images.githubusercontent.com/47882482/150610413-a45bf2fe-79e6-4262-995c-3c30845330ba.png)

Figure 8: Loss and Accuracy curves

(a) Training classification report | (b) Validation classification report
:-------------------------:|:-------------------------:
![image](https://user-images.githubusercontent.com/47882482/150610588-ce55b792-012b-4920-b81b-5ec61aa7d431.png) | ![image](https://user-images.githubusercontent.com/47882482/150610598-5fc89624-6649-4509-9254-2be94d010715.png)

Figure 9: First model classification reports
The first model showed a significant drop in accuracy values from training to
validation dataset.

### Second Model
After completing the first model, I created another model with two hidden layers
having 16 and 8 neurons respectively. In this model I used the feature of early
stopping which is it will automatically stop after 5 iteration (parameter which
can be set) if the validation loss doesn’t reduce. Below are the stats for the
model:

(a) Loss curves for train and validation | (b) Accuracy curves for train and validation
:-------------------------:|:-------------------------:
![image](https://user-images.githubusercontent.com/47882482/150610636-8156e86a-41e8-43f0-8eee-66138470b6d9.png) | ![image](https://user-images.githubusercontent.com/47882482/150610653-b6e8a1bd-f8de-467f-b942-c033a78e8ec3.png)

Figure 10: Loss and Accuracy curves

(a) Training classification report | (b) Validation classification report
:-------------------------:|:-------------------------:

![image](https://user-images.githubusercontent.com/47882482/150610699-6ec7e3a9-1137-48f4-9130-7bf699d6e83c.png) | ![image](https://user-images.githubusercontent.com/47882482/150610712-d8553cc6-bd5f-42cf-80d4-f2afd5193f0c.png)

Figure 11: Second model classification reports

Was able to do better than the first model, achieving training accuracy of 80%
and validation accuracy of 75%.

### Third Model
After completing the second model, same model architecture was used but added
l1 regularization on the second hidden layer. L1 regularization also known as
Lasso regularization add penalty which is the absolute value of the magnitude
of coecient. Below is the formula:

<img src="http://www.sciweavers.org/tex2img.php?eq=Loss%20%3D%20Error%28Y%20-%20%5Cwidehat%7BY%7D%29%20%2B%20%5Clambda%20%5Csum_1%5En%20%7Cw_i%7C%0A&bc=White&fc=Black&im=jpg&fs=18&ff=modern&edit=0" align="center" border="0" alt="Loss = Error(Y - \widehat{Y}) + \lambda \sum_1^n |w_i|" width="375" height="71" />

(a) Loss curves for train and validation | (b) Accuracy curves for train and validation
:-------------------------:|:-------------------------:
![image](https://user-images.githubusercontent.com/47882482/150610767-2c4dd55d-6591-46b3-954b-69ac7689ec44.png) | ![image](https://user-images.githubusercontent.com/47882482/150610782-67ccb9ca-0ab3-4184-9d36-3674144837b5.png)


Figure 12: Loss and Accuracy curves

(a) Training classification report | (b) Validation classification report
:-------------------------:|:-------------------------:

![image](https://user-images.githubusercontent.com/47882482/150610795-8fa4b1d3-0b18-4fea-a898-fd62db12c97f.png) | ![image](https://user-images.githubusercontent.com/47882482/150610805-00e892f4-6441-4214-84ec-c8efa20ccc24.png)

Figure 13: Third model classification reports

In this model I have achieved 79% and 75% on train and validation respectively.

### Fourth Model
After completing the second model, same model architecture was used but added
l1 regularization on the second hidden layer. L1 regularization also known as
Lasso regularization add penalty which is the absolute value of the magnitude
of coecient. Below is the formula:

<img src="http://www.sciweavers.org/tex2img.php?eq=Loss%20%3D%20Error%28Y%20-%20%5Cwidehat%7BY%7D%29%20%2B%20%20%5Clambda%20%5Csum_1%5En%20w_i%5E%7B2%7D%0A&bc=White&fc=Black&im=jpg&fs=18&ff=modern&edit=0" align="center" border="0" alt="Loss = Error(Y - \widehat{Y}) +  \lambda \sum_1^n w_i^{2}" width="358" height="71" />

(a) Loss curves for train and validation | (b) Accuracy curves for train and validation
:-------------------------:|:-------------------------:
![image](https://user-images.githubusercontent.com/47882482/150610975-cbfea71e-868d-4e4e-a75d-d37932579ef9.png) | ![image](https://user-images.githubusercontent.com/47882482/150610986-d17c5d61-7e62-4fe8-abf3-5b053c95e0ed.png)


Figure 14: Loss and Accuracy curves

(a) Training classification report | (b) Validation classification report
:-------------------------:|:-------------------------:
![image](https://user-images.githubusercontent.com/47882482/150611008-6110a719-3c2c-4e51-98e2-9f40979813f9.png) | ![image](https://user-images.githubusercontent.com/47882482/150611016-ec88c21d-cd7e-4729-89ee-3c29aa5dba24.png)

Figure 15: Fourth model classification reports

In this model I have achieved 80% and 75% on train and validation respectively.
We can see that the l2 regularization is working better than the l1 regularization
as the number of correlated variables are more in the dataset, therefore we see
l2 working better than l1.

### Fifth Model
The last model is the implementation of the dropout in the neural network
architecture. Dropout is also one of the regularization technique for neural
networks, in this we basically define the percentage of output nodes we want to
drop from our calculation into the next layer.


(a) Loss curves for train and validation | (b) Accuracy curves for train and validation
:-------------------------:|:-------------------------:
![image](https://user-images.githubusercontent.com/47882482/150611060-72af105b-0aab-470a-9bf6-cee7635d61d2.png) | ![image](https://user-images.githubusercontent.com/47882482/150611068-824f659d-7add-4ad1-ae56-af55f9ad2d5b.png)


Figure 16: Loss and Accuracy curves

(a) Training classification report | (b) Validation classification report
:-------------------------:|:-------------------------:
![image](https://user-images.githubusercontent.com/47882482/150611080-1545d164-6c3d-419d-9964-66d57204d5d0.png) | ![image](https://user-images.githubusercontent.com/47882482/150611086-42fa231e-9d76-4f86-9adb-ed5ba22ca21b.png)

Figure 17: Fifth model classification reports
In this model I have achieved 80% and 75% on train and validation respectively.
We can see that the l2 regularization is working better than the l1 regularization
as the number of correlated variables are more in the dataset, therefore we see
l2 working better than l1.

### Test Results
In the end the fourth model which was the neural network model with l2 regu-
larization worked the best as the both the learning curve and accuracy curves
where in line for both the train and the validation set. Hence I went forward
with that and achieved an accuracy of 77% on the dataset.

![image](https://user-images.githubusercontent.com/47882482/150611098-529cac3c-76c7-44d1-be98-1803f089765e.png)

Figure 18: This is an image showing classification report on the test set.

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

<img src="https://render.githubusercontent.com/render/math?math=e^{i \pi} = -1">
<img src="https://render.githubusercontent.com/render/math?math=z =x_i%2Bu / \sigma">

Further the dataset was divided into 3 parts: Train, Validation and Test. I used
stratified splitting and the split size was 60-20-20 (train-validation-test).

4LogisticRegression
Logistic regression was coded in python using the below set of equations:
Below equation multiples the weight with each features and add bias to it

z=wT.X+b (1)
After calculating the above step we pass it on to the sigmoid function

f(z)=
1
1+ez
=
ez
1+ez
(2)
Once the value is returned from the sigmoid function we check for the loss using
the equation

j(✓)=
1
m
Xm
i=
⇥
y(i)log(h✓(x(i))) + (1y(i)) log(1h✓(x)(i))
⇤
(3)
On the basis of loss we update our weights and bias using:

w=w↵⇤dw (4)
b=b↵⇤db (5)
4.1 Results
Two logistic regression models were created having learning rate as 0.05 (Model
A) and 0.01 (Model B). Both the model ran for the same number of iterations
which was 1000. Below are loss curves for the same.

Figure 1: This is an image showing the classification report when learning rate
is 0.05.

Figure 2: This is an image showing the classification report when learning rate
is 0.01.

From the image 2 it can be seen that model with learning rate 0.01 hasn’t
converged and can be trained on more epochs.
Model A achieved an accuracy of 78% on the train set and 77% on the validation
set whereas Model B achieved an accuracy of 78% on train and 76% on the
validation set.

Below are the classification report of the above two models:

Figure 3: This is an image shows the loss curve when learning rate is 0.05.
Figure 4: This is an image shows the loss curve when learning rate is 0.01.
With the both the models performing almost the same, I have chosen model
with learning rate 0.05 as it has better accuracy on the validation set.

4.2 Test Result
On the test data the model achieved an accuracy of 79%. Below is the classifi-
cation report on the test data:

Figure 5: This is an image shows the loss curve when learning rate is 0.01.
5 Neural Networks
Neural network is a set of system which was inspired by biological neural net-
work also know as brain. They are a collection of neurons, hidden layers and
activation function. Inside a neural network there are two type of passes known
as forward pass or forward propagation and backward pass or backward propa-
gation.

5.1 Forward Propagation
In forward propagation we assign weights to each and every connection between
neurons. Each layer has it’s own set of activation function which can vary
based on each and every hidden layer. Below are the steps involved in forward
propagation:

z=wT.X+b (6)
The above equation will assign weights to each neuron connection and one bias
per layer. Once this is done we can apply di↵erent type of activation function
like Sigmoid or Relu.
Sigmoid activation function equation:

y=
1
1+ez
=
ez
1+ez
(7)
Figure 6: This is an image showing sigmoid activation function.
Relu activation function equation:

y=
(
1 , ifz 0 ,
0 , otherwise,
(8)
Figure 7: This is an image showing relu activation function.
There are many more activation function which could be used like tanh, leaky
relu, etc but the above two are ones which are used in my model building process.

5.2 Backward Propagation
Once we have completed the forward propagation we need to compute the loss
and try to minimize the loss by updating weights and bias for each and every
neuron connection, exactly what is done in backward propagation. Loss was
calculated in the last layer having just one neuron by using binary cross-entropy.
Below is the formula:

j(✓)=
1
m
Xm
i=
⇥
y(i)log(h✓(x(i))) + (1y(i)) log(1h✓(x)(i))
⇤
(9)
On the basis of loss we update our weights for each and every neuron connection
and bias for each hidden layer using:

w=w↵⇤dw (10)
b=b↵⇤db (11)
The above whole process comes under backward propagation. A complete cycle
of forward propagation and backward propagation is known has one epoch.

5.3 First Model
I have trained my first model by taking only one hidden layer and inside that
hidden layer took 32 neurons. For all the models build a standard learning rate
of 0.001 and batch size of 64 was taken. Below are the training and validations
loss and accuracy curves:

(a) Loss curves for train and validation (b) Accuracy curves for train and
validation
Figure 8: Loss and Accuracy curves
(a) Training classification report (b) Validation classification report
Figure 9: First model classification reports
The first model showed a significant drop in accuracy values from training to
validation dataset.

5.4 Second Model
After completing the first model, I created another model with two hidden layers
having 16 and 8 neurons respectively. In this model I used the feature of early
stopping which is it will automatically stop after 5 iteration (parameter which
can be set) if the validation loss doesn’t reduce. Below are the stats for the
model:

(a) Loss curves for train and validation (b) Accuracy curves for train and
validation
Figure 10: Loss and Accuracy curves
(a) Training classification report (b) Validation classification report
Figure 11: Second model classification reports
Was able to do better than the first model, achieving training accuracy of 80%
and validation accuracy of 75%.

5.5 Third Model
After completing the second model, same model architecture was used but added
l1 regularization on the second hidden layer. L1 regularization also known as
Lasso regularization add penalty which is the absolute value of the magnitude
of coecient. Below is the formula:

Loss=Error(YYb)+
Xn
1
|wi| (12)
(a) Loss curves for train and validation (b) Accuracy curves for train and
validation
Figure 12: Loss and Accuracy curves
(a) Training classification report (b) Validation classification report
Figure 13: Third model classification reports
In this model I have achieved 79% and 75% on train and validation respectively.

5.6 Fourth Model
After completing the second model, same model architecture was used but added
l1 regularization on the second hidden layer. L1 regularization also known as
Lasso regularization add penalty which is the absolute value of the magnitude
of coecient. Below is the formula:

Loss=Error(YYb)+
Xn
1
wi^2 (13)
(a) Loss curves for train and validation (b) Accuracy curves for train and
validation
Figure 14: Loss and Accuracy curves
(a) Training classification report (b) Validation classification report
Figure 15: Fourth model classification reports
In this model I have achieved 80% and 75% on train and validation respectively.
We can see that the l2 regularization is working better than the l1 regularization
as the number of correlated variables are more in the dataset, therefore we see
l2 working better than l1.

5.7 Fifth Model
The last model is the implementation of the dropout in the neural network
architecture. Dropout is also one of the regularization technique for neural
networks, in this we basically define the percentage of output nodes we want to
drop from our calculation into the next layer.

(a) Loss curves for train and validation (b) Accuracy curves for train and
validation
Figure 16: Loss and Accuracy curves
(a) Training classification report (b) Validation classification report
Figure 17: Fifth model classification reports
In this model I have achieved 80% and 75% on train and validation respectively.
We can see that the l2 regularization is working better than the l1 regularization
as the number of correlated variables are more in the dataset, therefore we see
l2 working better than l1.

5.8 Test Results
In the end the fourth model which was the neural network model with l2 regu-
larization worked the best as the both the learning curve and accuracy curves
where in line for both the train and the validation set. Hence I went forward
with that and achieved an accuracy of 77% on the dataset.

Figure 18: This is an image showing classification report on the test set.

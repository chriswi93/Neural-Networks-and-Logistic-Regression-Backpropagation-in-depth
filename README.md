<h1>Neural Networks and Logistic Regression - Backpropagation in depth</h1>
<p align="left" style="font-size:13px;"><b>Christoph Winkler</b><br>
<i>M. Sc. Business Information Systems, Data Scientist.<br>
Generative Modeling is simply about modeling "How the world could be" and not necessarily "How the world actually is".</i></p>
<br>
Logistic regression is one of the most popular machine learning models for classification. Mathematically, logistic regression is a special case of a neural network and is therefore an appropriate introduction to learn how backpropagation in neural networks actually works. In this article the theoretical foundation on how to derive the equations for optimization of the model parameters is discussed. In particular, the derivation of the equations for gradient descent is shown. Furthermore, a Python implementation is presented and model convergence is analyzed based on a synthetic sample data set. This article is written for practitioners and researchers that have a basic understanding of neural networks, calculus and optimization.<br>

<h2>Model</h2>
In general, logistic regression is a classification model that learns to generalize from a training data set x if a new data point either belongs to class zero or class one. It is also important to note that logistic regression can only predict probabilities between zero and one and therefore can only be used for classification and not for regression. In comparison to neural networks, logistic regression also has multiple input nodes which represent the features of a data set. Additionally, logistic regression has no hidden layer and only a single output node with a sigmoid activation function. The sigmoid activation function is applied to transform any linear combination z = wx + b without any upper or lower bound into log odds of two classes which equals the probability g that input x belongs to class zero or one. If the weight tensor w that equals the connection weights between input nodes and output node takes on a large activation for x, the probability for class one is also quite large. Moreover, logistic regression is a linear and discriminative model. It therefore learns to predict P(y = 1|x) to distinguish between two classes.

<p align="center">
<img src="logistic_regression.png"/>
</p>

<h2>Loss and Optimization</h2>
The loss that is minimized during training is known as binary cross entropy. It decreases if the predicted probability for the class labels g get closer to the true class labels y and is therefore an appropriate measure to monitor the learning progress and convergence of the model (see illustration below). 
<p align="center">
<img src="loss.png"/>
</p>
In general, binary cross entropy is a measure of uncertantity. Since we know the true distribution p(y), but not the function to distinguish between the two classes, we would like to learn a linear combination with an outcome distribution q(y) that approximates the outcome of true distribution p(y) as much as possible. This is how classification in neural networks and logistic regression actually works. The only difference between neural networks and logistic regression is that neural networks are able to model non linear relationships due to hidden layers and non linear activations which is not possible in a linear model like logistic regression.
<br><br>
Now let's discuss the steps to make predictions (alias forward propagation) and propagate back the error to adapt the model parameters w and b based on x. First the data set x is forward propagated through the model. The forward propagation step in logistic regression is composed of three steps.

<p align="center">
<a href="https://www.codecogs.com/eqnedit.php?latex=\dpi{120}&space;\newline&space;z&space;=&space;wx^{T}&space;&plus;&space;b&space;\newline&space;g&space;=&space;\sigma&space;(z)&space;=&space;\frac{1}{1&plus;e^{-z}}&space;\newline&space;L&space;=&space;-\sum_{i}^{m}&space;y\log(g)&space;&plus;&space;(1&space;-&space;y)\log(1-g)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\dpi{120}&space;\newline&space;z&space;=&space;wx^{T}&space;&plus;&space;b&space;\newline&space;g&space;=&space;\sigma&space;(z)&space;=&space;\frac{1}{1&plus;e^{-z}}&space;\newline&space;L&space;=&space;-\sum_{i}^{m}&space;y\log(g)&space;&plus;&space;(1&space;-&space;y)\log(1-g)" title="\newline z = wx^{T} + b \newline g = \sigma (z) = \frac{1}{1+e^{-z}} \newline L = -\sum_{i}^{m} y\log(g) + (1 - y)\log(1-g)" /></a>
</p>

We compute the outcome z of a linear function by multiplying data x with a weight tensor w and add a bias term b. Aftwards we transform the linear combination z into probabilities using the the sigmoid activation function which returns the predictions g. The predictions g are plugged into the binary cross entropy loss function L. <br><br>
Afterwards the error is computed which is required to propagate back the error to w and b. In order to improve the predictive power of the model we have to adapt the weights w and the bias term b according to the computed error. The optimization technique known as gradient descent helps us to find out in which direction we have to adapt the model parameters. In particular, in gradient descent we compute the gradients of the variables w and b on L to determine the direction of the steepest descent on the loss function. Additionally, we specify a learning rate that represents the step size on the loss function and is usually smaller than one. It is multiplied by the gradients and is therefore a scaling factor to control learning speed in each iteration. If the learning rate is chosen very large the learning speed increases, but the algorithm might jump over the global minimum and slowly converges. In contrast, if the learning rate is chosen very small the learning speed is very slow and the computational efforts and training time for large data sets increase. Therefore, it is important to choose the learning rate carefully.

<h2>Computation Graph</h2>
In order to use gradient descent we have to derive the equations for the gradients on the loss function. This is best illustrated using a so called computation graph. Computation graphs are widely used in mathematical applications to discover dependencies which help us to derive the equations for the gradients. In the context of optimization a computation graph is the application of the chain rule in calculus.
<br>
<p align="center">
<img src="computation_graph.png"/>
</p>

<p align="center">
<a href="https://www.codecogs.com/eqnedit.php?latex=\dpi{120}&space;\frac{\partial&space;L}{\partial&space;w}&space;=&space;\frac{\partial&space;L}{\partial&space;g}\frac{\partial&space;g}{\partial&space;z}\frac{\partial&space;z}{\partial&space;w}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\dpi{120}&space;\frac{\partial&space;L}{\partial&space;w}&space;=&space;\frac{\partial&space;L}{\partial&space;g}\frac{\partial&space;g}{\partial&space;z}\frac{\partial&space;z}{\partial&space;w}" title="\frac{\partial L}{\partial w} = \frac{\partial L}{\partial g}\frac{\partial g}{\partial z}\frac{\partial z}{\partial w}" /></a>
</p>

If we want to compute the gradient of a dependent variable w or b on L, we have to go back the computation graph and multiply the gradients by each other.

<h2>Derivatives for Gradient Descent</h2>

So, let's compute the first derivative of the loss beginning with the first node on the right.

<p align="center">
<a href="https://www.codecogs.com/eqnedit.php?latex=\dpi{120}&space;\frac{\partial&space;L}{\partial&space;g}&space;=&space;\frac{1-y}{1-g}&space;-&space;\frac{y}{g}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\dpi{120}&space;\frac{\partial&space;L}{\partial&space;g}&space;=&space;\frac{1-y}{1-g}&space;-&space;\frac{y}{g}" title="\frac{\partial L}{\partial g} = \frac{1-y}{1-g} - \frac{y}{g}" /></a>
</p>

Next we have to compute the derivative of the simgoid function. This step is a bit more complex if you are not familiar with calculus.

<p align="center">
<a href="https://www.codecogs.com/eqnedit.php?latex=\dpi{120}&space;\newline\frac{\partial&space;g}{\partial&space;z}&space;=&space;\frac{e^{-z}}{(e^{-z}&space;&plus;&space;1)^2}&space;=&space;\frac{1}{e^{-z}&space;&plus;&space;1}\frac{e^{-z}}{e^{-z}&space;&plus;&space;1}&space;\newline&space;\frac{\partial&space;g}{\partial&space;z}&space;=&space;\frac{1}{e^{-z}&space;&plus;&space;1}\frac{(e^{-z}&space;&plus;&space;1)&space;-&space;1}{e^{-z}&space;&plus;&space;1}&space;\newline&space;\frac{\partial&space;g}{\partial&space;z}&space;=&space;\frac{1}{e^{-z}&space;&plus;&space;1}\left&space;(\frac{e^{-z}&space;&plus;&space;1}{e^{-z}&space;&plus;&space;1}-\frac{1}{e^{-z}&space;&plus;&space;1}&space;\right&space;)&space;\newline&space;\frac{\partial&space;g}{\partial&space;z}&space;=&space;\frac{1}{e^{-z}&space;&plus;&space;1}\left&space;(1-\frac{1}{e^{-z}&space;&plus;&space;1}&space;\right&space;)&space;=&space;\sigma&space;(z)(1-\sigma&space;(z))" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\dpi{120}&space;\newline\frac{\partial&space;g}{\partial&space;z}&space;=&space;\frac{e^{-z}}{(e^{-z}&space;&plus;&space;1)^2}&space;=&space;\frac{1}{e^{-z}&space;&plus;&space;1}\frac{e^{-z}}{e^{-z}&space;&plus;&space;1}&space;\newline&space;\frac{\partial&space;g}{\partial&space;z}&space;=&space;\frac{1}{e^{-z}&space;&plus;&space;1}\frac{(e^{-z}&space;&plus;&space;1)&space;-&space;1}{e^{-z}&space;&plus;&space;1}&space;\newline&space;\frac{\partial&space;g}{\partial&space;z}&space;=&space;\frac{1}{e^{-z}&space;&plus;&space;1}\left&space;(\frac{e^{-z}&space;&plus;&space;1}{e^{-z}&space;&plus;&space;1}-\frac{1}{e^{-z}&space;&plus;&space;1}&space;\right&space;)&space;\newline&space;\frac{\partial&space;g}{\partial&space;z}&space;=&space;\frac{1}{e^{-z}&space;&plus;&space;1}\left&space;(1-\frac{1}{e^{-z}&space;&plus;&space;1}&space;\right&space;)&space;=&space;\sigma&space;(z)(1-\sigma&space;(z))" title="\newline\frac{\partial g}{\partial z} = \frac{e^{-z}}{(e^{-z} + 1)^2} = \frac{1}{e^{-z} + 1}\frac{e^{-z}}{e^{-z} + 1} \newline \frac{\partial g}{\partial z} = \frac{1}{e^{-z} + 1}\frac{(e^{-z} + 1) - 1}{e^{-z} + 1} \newline \frac{\partial g}{\partial z} = \frac{1}{e^{-z} + 1}\left (\frac{e^{-z} + 1}{e^{-z} + 1}-\frac{1}{e^{-z} + 1} \right ) \newline \frac{\partial g}{\partial z} = \frac{1}{e^{-z} + 1}\left (1-\frac{1}{e^{-z} + 1} \right ) = \sigma (z)(1-\sigma (z))" /></a>
</p>

Last step is quite simple. We have to derive the partial derivatives for w and b.

<p align="center">
<a href="https://www.codecogs.com/eqnedit.php?latex=\dpi{120}&space;\newline\frac{\partial&space;z}{\partial&space;w}&space;=&space;x^{T}\newline&space;\frac{\partial&space;z}{\partial&space;b}&space;=&space;1" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\dpi{120}&space;\newline\frac{\partial&space;z}{\partial&space;w}&space;=&space;x^{T}\newline&space;\frac{\partial&space;z}{\partial&space;b}&space;=&space;1" title="\newline\frac{\partial z}{\partial w} = x^{T}\newline \frac{\partial z}{\partial b} = 1" /></a>
</p>

Now, we only have to multiply the gradients by each other, take the sum of the gradients for x with sample size m and multiply it by the inverse of m to compute the average gradients w and b for x.

<p align="center">
<a href="https://www.codecogs.com/eqnedit.php?latex=\dpi{120}&space;\newline\frac{\partial&space;L}{\partial&space;w}&space;=&space;\frac{1}{m}\sum_{i}^{m}\frac{\partial&space;L}{\partial&space;g}\frac{\partial&space;g}{\partial&space;z}\frac{\partial&space;z}{\partial&space;w}\newline\frac{\partial&space;L}{\partial&space;b}&space;=&space;\frac{1}{m}\sum_{i}^{m}\frac{\partial&space;L}{\partial&space;g}\frac{\partial&space;g}{\partial&space;z}\frac{\partial&space;z}{\partial&space;b}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\dpi{120}&space;\newline\frac{\partial&space;L}{\partial&space;w}&space;=&space;\frac{1}{m}\sum_{i}^{m}\frac{\partial&space;L}{\partial&space;g}\frac{\partial&space;g}{\partial&space;z}\frac{\partial&space;z}{\partial&space;w}\newline\frac{\partial&space;L}{\partial&space;b}&space;=&space;\frac{1}{m}\sum_{i}^{m}\frac{\partial&space;L}{\partial&space;g}\frac{\partial&space;g}{\partial&space;z}\frac{\partial&space;z}{\partial&space;b}" title="\newline\frac{\partial L}{\partial w} = \frac{1}{m}\sum_{i}^{m}\frac{\partial L}{\partial g}\frac{\partial g}{\partial z}\frac{\partial z}{\partial w}\newline\frac{\partial L}{\partial b} = \frac{1}{m}\sum_{i}^{m}\frac{\partial L}{\partial g}\frac{\partial g}{\partial z}\frac{\partial z}{\partial b}" /></a>
</p>

Last step of gradient descent is to scale the gradients by the learning rate and change the variables in the direction of the computed gradients.

<p align="center">
<a href="https://www.codecogs.com/eqnedit.php?latex=\dpi{120}&space;\newline&space;w&space;=&space;w&space;-&space;\frac{\partial&space;L}{\partial&space;w}&space;*&space;learningrate&space;\newline&space;b&space;=&space;b&space;-&space;\frac{\partial&space;L}{\partial&space;b}&space;*&space;learningrate" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\dpi{120}&space;\newline&space;w&space;=&space;w&space;-&space;\frac{\partial&space;L}{\partial&space;w}&space;*&space;learningrate&space;\newline&space;b&space;=&space;b&space;-&space;\frac{\partial&space;L}{\partial&space;b}&space;*&space;learningrate" title="\newline w = w - \frac{\partial L}{\partial w} * learningrate \newline b = b - \frac{\partial L}{\partial b} * learningrate" /></a>
</p>

In practice forward propagation and backpropagation are repeated several times. It is guaranteed for logistic regression that the parameters converge to the global minimum. However this does not apply for a neural network that approximates a function with non linear relationships. In this case the convergence is only guaranteed to a local minimum, because non linear functions are always non convex. This is one reason why a linear model should be preferred when a data set is linearly separable. Below you can find an implementation in Python on how to compute the gradients and update the model parameters of logistic regression.<br>

```python
def predict(self, x):
  z = np.matmul(self.w, x.T) + self.b
  g = sigmoid(z)
  return z, g
  
def compute_gradients(self, x, y):
  # forward pass to get logits (z) and probability (g) values
  z, g = self.predict(x)
        
  # compute gradients
  dL_dg = (1 - y) / (1 - g) - y / g
  dg_dz = g * (1 - g)
  dz_w = x.T
        
  # apply chain rule and compute sum of gradients in batch
  dL_dw = np.sum(dL_dg * dg_dz * dz_w, axis=1)
  dL_db = np.sum(dL_dg * dg_dz, axis=1)
        
  # compute average of gradients in batch
  dL_dw /= len(y)
  dL_db /= len(y)
        
  return dL_dw, dL_db
  
def optimize(self, dL_dw, dL_db, learning_rate):
  self.w = self.w - dL_dw * learning_rate
  self.b = self.b - dL_db * learning_rate
```

Now, let's create a random data set with 100,000 samples and two different classes generated by two bivariate normal distributions with μ = (0,0) and μ = (2,8) without any correlation between x1 and x2. Afterwards, two models are trained for 100 epochs what means that each model can see each data point 100 times and the model parameters are updated 100 times. On the left side the model is trained on a sample data set that is not centered to μ = 0 and on the right side the model is trained on a sample data set that is centered to μ = 0. Below you can see how centering data to zero can affect convergence of logistic regression and neural networks. This is illustrated below. The darker the area in the image the more confident the model is about the class of a data point in this area.
<p align="center">
<img src="classification_sample.png"/>
</p>

As you can see the convergence speed of the model trained on centered data with μ = 0 is much higher. Additionally, you can compare the accuracy and the loss during training measured on a hold out test set below (20 % split).

<p align="center">
<img src="loss_sample.png"/>
</p>

It confirms that centering the data greatly improves the convergence speed of the model which is expressed by higher accuracy and lower loss. Therefore, I would always recommend to normalize or scale your data set. It can greatly affect the performance of your model. <br><br>
That's all you need to know about gradient descent and backpropagation. There are many extensions of gradient descent that are not discussed in this article. In the next article we will discuss one of most popular extensions called stochastic gradient descent. Thank you for reading my article! I hope it helps you to get a better understanding on how backpropagation in neural networks actually works.

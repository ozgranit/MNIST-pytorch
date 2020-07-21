# MNIST-pytorch
playing with pytorch neural networks to solve MNIST handwrriten digit classification

mnist_1.py - Naive logistic regression model

mnist_2.py - Improved parameters: batch_size=200, learning_rate=0.001, with Adam algorithm as the optimizer.

mnist_3.py - A deeper model using a ReLU non-linear layer and another linear layer.
The size of the hidden layer is 500.

mnist.py - running "python mnist.py" will prouduce plots of training loss and accuracy (uses mnist_1.py, mnist_2.py, mnist_3.py) of all three models.

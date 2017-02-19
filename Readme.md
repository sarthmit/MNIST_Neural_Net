Self-made implementation of training of a Neural Network on the MNIST dataset.

Language used: Octave

Mnist_Neural_Net.m - Master program to unload the data and then call respective functions.
Mnist_data.mat - contains preprocessed data.
Sarthak_display.m - displays random elements of the dataset.
Sarthak_cost.m - returns the cost calculated and the gradients.
Sarthak_forward.m - feed forward to get the prediction.
Sarthak_gradientDescent.m - Applies gradient descent using the gradients calculated in Sarthak_cost.m to reach a minima.
Sarthak_predict.m - returns the predicted values using the final obtained parameters.
Sarthak_sigmoid.m - calculates the sigmoid of a given matrix/vector.

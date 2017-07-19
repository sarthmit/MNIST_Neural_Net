clear; close all; clc

% 0 is mapped to 10 everywhere
% X is a matrix of features, y of their labels.

load('Mnist_data.mat');

[m n] = size(X);

% ----------------Display Data -------------------

random = randperm(size(X, 1));
random = random(1:100);

disp('Displaying Part of dataset');

Sarthak_display(X(random,:));

pause;

% --------------Train Neural Network --------------

num_labels = 10;
hidden_layer = 50;
alpha = 3;
lambda = 1;
max_iter = 1000;
epsilon = 0.25;

Theta1 = rand(hidden_layer,n+1)*2*epsilon - epsilon;
Theta2 = rand(num_labels, hidden_layer+1)*2*epsilon-epsilon;

parameters = [Theta1(:);Theta2(:)];

parameters = Sarthak_gradientDescent(X,y,parameters, num_labels, hidden_layer, alpha, max_iter, lambda);

pause;

%-------------------- Prediction -----------------

s = Sarthak_predict(X, parameters, num_labels, hidden_layer);
s = mean(double(s==y))*100;
fprintf('The accuracy of Neural Network is: %f', s);
disp(' ');
pause;
function [a1,a2,a3] = Sarthak_forward(X, parameters, num_labels, hidden_layer)

[m n] = size(X);

Theta1 = reshape(parameters(1:hidden_layer*(n+1)), hidden_layer, n+1);

Theta2 = reshape(parameters(hidden_layer*(n+1)+1:end), num_labels, hidden_layer+1);

a1 = [ones(m,1) X];
z2 = a1 * Theta1';
a2 = Sarthak_sigmoid(z2);
a2 = [ones(m,1) a2];
z3 = a2 * Theta2';
a3 = Sarthak_sigmoid(z3);

end
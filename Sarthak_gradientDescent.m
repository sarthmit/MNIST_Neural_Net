function [parameters] = Sarthak_gradientDescent(X,Y,parameters, num_labels, hidden_layer, alpha, max_iter, lambda)

[m n] = size(X);

Theta1 = reshape(parameters(1:hidden_layer*(n+1)), hidden_layer, n+1);

Theta2 = reshape(parameters(hidden_layer*(n+1)+1:end), num_labels, hidden_layer+1);

for i=1:max_iter
	[cost grad] = Sarthak_cost(X,Y,parameters,num_labels,hidden_layer,lambda);
	grad_Theta1 = reshape(grad(1:hidden_layer*(n+1)), hidden_layer, n+1);
	grad_Theta2 = reshape(grad(hidden_layer*(n+1)+1:end), num_labels, hidden_layer+1);
	fprintf('Iteration : %d  |  Cost : %f\n', i, cost);
	plot(i,cost);
	hold on;
	Theta1 = Theta1 - alpha * grad_Theta1;
	Theta2 = Theta2 - alpha * grad_Theta2;
	parameters = [Theta1(:);Theta2(:)];
end
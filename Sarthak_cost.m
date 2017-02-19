function [cost grad] = Sarthak_cost(X,Y,parameters,num_labels,hidden_layer,lambda)

[m n] = size(X);

Theta1 = reshape(parameters(1:hidden_layer*(n+1)), hidden_layer, n+1);

Theta2 = reshape(parameters(hidden_layer*(n+1)+1:end), num_labels, hidden_layer+1);

cost = 0;

[a1 a2 a3] = Sarthak_forward(X,parameters, num_labels, hidden_layer);


for i =1:num_labels
	s=Y==i;
	cost = cost - sum(s.*log(a3(:,i)) + (1-s).*log(1-a3(:,i)))/m;
end

cost = cost + (sum(sum(Theta1(:,2:end).^2)) + sum(sum(Theta2(:,2:end).^2)))*lambda/(2*m);

s= [1:num_labels]';
del_1 = zeros(hidden_layer,n+1);
del_2 = zeros(num_labels, hidden_layer+1);
for i = 1:m
	p=s==Y(i);
	delta_3 = a3(i,:)' - p;
	delta_2 = (Theta2' * delta_3)(2:end).*((a2(i,:).*(1-a2(i,:)))(2:end))';
	del_2 = del_2 + delta_3 * a2(i,:);
	del_1 = del_1 + delta_2 * a1(i,:);
end

Theta1_grad= del_1/m;
Theta2_grad=del_2/m;

Theta1_grad(:,2:end) = Theta1_grad(:,2:end) + lambda/m * Theta1(:,2:end);
Theta2_grad(:,2:end) = Theta2_grad(:,2:end) + lambda/m * Theta2(:,2:end);

grad = [Theta1_grad(:); Theta2_grad(:)];
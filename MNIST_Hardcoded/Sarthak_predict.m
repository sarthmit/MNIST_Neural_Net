function S = Sarthak_predict(X, parameters, num_labels, hidden_layer)

[m n] = size(X);

Theta1 = reshape(parameters(1:hidden_layer*(n+1)), hidden_layer, n+1);

Theta2 = reshape(parameters(hidden_layer*(n+1)+1:end), num_labels, hidden_layer+1);

S = zeros(m,1);

[a1 a2 a3] = Sarthak_forward(X, parameters, num_labels, hidden_layer);

for i=1:m
	S(i) = find(a3(i,:)==max(a3(i,:)));
end
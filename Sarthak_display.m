function[] = Sarthak_display(X)

num_examples = size(X,1);

length = sqrt(num_examples);
width = length;

pad = 2;
feature_square_size = sqrt(size(X,2));

curr_x = pad;
curr_y = pad;

display_arr = - ones(pad + length*(pad+feature_square_size), pad + width*(pad+feature_square_size));

for i=1:num_examples
	S = reshape(X(i,:),feature_square_size,feature_square_size);
	display_arr(curr_x+1:curr_x+feature_square_size, curr_y+1:curr_y+feature_square_size) = S;
	if(mod(i,10)==0)
		curr_x = curr_x+pad+feature_square_size;
		curr_y=pad;
	else
		curr_y = curr_y+feature_square_size + pad;
	end
end

colormap(gray);
imagesc(display_arr, [-1 1]);
axis image off;

end;
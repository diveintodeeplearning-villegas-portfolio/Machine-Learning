function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%
%5000 x 400
NewCol=ones(m,1);
a1 = [NewCol X];
% 5000x401 x 401x25 
z2=a1*transpose(Theta1);
%5000x25
a2=sigmoid(z2);
a2=[NewCol a2];
%5000x26 x 26x10
z3=a2*transpose(Theta2);
%5000x10
a3=sigmoid(z3);

yv=[1:num_labels] == y;

all_combos = eye(num_labels);    
y_matrix = all_combos(y,:)        % works for Matlab

J=(1/m)*(sum(sum((((-yv).*log(a3))-((1-yv).*(log(1-a3)))))));

%first col to zero

Theta1(:,1)=0;
Theta2(:,1)=0;

reg=(lambda/(2*m))*(sum(sum(Theta1.^2))+sum(sum(Theta2.^2)));

J=J+reg;

%5000x10 - 5000x1
delta3=a3-y_matrix;
% 5000x10 * 10x26 .* 5000x26  = 26x5000
temp=transpose(delta3*Theta2(:,2:end));
%fprintf('size(temp) is [%s]\n', int2str(size(temp)));
%fprintf('size(grad) is [%s]\n', int2str(size(sigmoidGradient(z2))));
delta2=transpose(temp).* sigmoidGradient(z2);
%fprintf('size(delta2) is [%s]\n', int2str(size(delta2)));
%delta2 = delta2(2:end);

%25x401 = 25x401 +25x5000 * 5000x401
%fprintf('size(theta1_grad) is [%s]\n', int2str(size(Theta1_grad)));
%fprintf('size(delta2) is [%s]\n', int2str(size(delta2)));
%fprintf('size(a1) is [%s]\n', int2str(size(a1)));

Theta1_grad=Theta1_grad+ transpose(delta2)*a1;
%10x26 = 10x26 + 10x5000 * 5000*26
Theta2_grad=Theta2_grad+ transpose(delta3)*a2;

Theta1_grad=Theta1_grad/m +(lambda/m)*Theta1;
Theta2_grad=Theta2_grad/m +(lambda/m)*Theta2;


% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end

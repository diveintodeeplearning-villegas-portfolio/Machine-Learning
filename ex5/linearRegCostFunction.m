function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%


hx=X*theta;
temp=(hx-y);

theta2=zeros(size(theta));
theta2(2:end)=theta(2:end);

J= ((1/(2*m)).*(sum(temp.^2)))+((lambda/(2*m)).*sum(theta2.^2));
J=sum(J);

fprintf('size(temp) is [%s]\n', int2str(size(temp)));
fprintf('size(X) is [%s]\n', int2str(size(X)));

grad=(1/m)*(transpose(temp)*X);
reg= ((lambda/m)*transpose(theta2));
reg(1)=0;
grad=grad+reg;








% =========================================================================

grad = grad(:);

end

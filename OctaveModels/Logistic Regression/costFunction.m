function [cost,grad] = costFunction(x,y,theta)
 
  m = length(y);
  h_of_x = sigmoid(x * theta);
  
cost=0;
grad = zeros(size(theta));

cost = 1 / m * sum( -1 * y' * log(h_of_x) - (1-y') * log(1 - h_of_x) );
grad = 1 / m * (x' * (h_of_x - y));
end
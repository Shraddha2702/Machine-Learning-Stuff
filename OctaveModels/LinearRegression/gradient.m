function [theta,JHistory] = gradient(x,y,alpha,theta,iterations)

m = length(y);
JHistory = zeros(m,1);

for i = 1:iterations
  t1 = theta(1) - alpha/m * sum((((x*theta)-y).*x(:,1)));
  t2 = theta(2) - alpha/m * sum((((x*theta)-y).*x(:,2)));
  
  theta(1) = t1;
  theta(2) = t2;
 
  JHistory(i) = computeCost(x,y,theta);
end
end
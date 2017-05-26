function [theta,cost] = computeTheta(x,y)

cost = 0;
m = length(x);
theta = zeros(m,1);

theta = pinv(x'*x)*x'*y;
cost = computeCost(x,y,theta);
end
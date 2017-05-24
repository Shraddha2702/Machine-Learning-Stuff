function J = computeCost(x,y,theta)

m = length(y);

J = 0;

J = 1/(2*m) * sum(((x*theta)-y).^2);

end
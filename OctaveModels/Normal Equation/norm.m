data = load('ex1data2.txt');

X = data(:,1:2);
Y = data(:,3);

[X, mu, st] = normalize(X);

plot(X,Y,'ro')
xlabel('x')
ylabel('y');
hold on;

m = length(X);

X = [ones(m,1) X(:,1:2)];

r = size(X,2);

theta = computeTheta(X,Y,theta);

plot(X(:,2),X*theta,'-')
legend('training data','normalize');
hold off;
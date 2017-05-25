data = load('ex2data1.txt');
X = data(:,1:2);
Y = data(:,3);

pos = find(Y == 1);
neg = find(Y == 0);

plot(X(pos, 1), X(pos, 2), 'k+','LineWidth', 50, ...
'MarkerSize', 7)
plot(X(neg, 1), X(neg, 2), 'ko', 'MarkerFaceColor', 'y', ...
'MarkerSize', 7);
xlabel('Subject 1 Marks');
ylabel('Subject 2 Marks');

[m n] = size(X);

theta  = zeros(n+1,1);
X = [ones(m,1) X];

[cost,grad] = costFunction(X,Y,theta);

printf('cost : %d',cost);
clc;

% Plotting of the whole data first
d = load('test.txt');
X=d(:,1);
Y=d(:,2);
plotData(X,Y);
printf('\nDone plotting data');

theta = zeros(2,1);

m = length(Y);
xt = [ones(m,1) d(:,1)];
computeCost(xt,Y,theta);

% Find theta revised on the basis of cost till some
%particular no of iterations  ie 100 here and get the min value of theta 
alpha = 0.01;
iterations = 100;

theta = gradient(xt,Y,alpha,theta,iterations);

fprintf('\nTheta found by gradient descent');
fprintf('%d %d',theta(1),theta(2));

%plot graphs with same values of theta
hold on;
plot(xt(:,2),xt*theta,'-');
legend('Training data','linear regression');
hold off;

printf('\nProgram paused, Press enter to continue');
pause;


%Visualizing Jtheta

J_theta0 = linspace(-10,10,100);
J_theta1 = linspace(-1,4,100);

Jvals = [length(J_theta0),length(J_theta1)];

for i = 1:length(J_theta0)
  for j = 1:length(J_theta1)
    thetaj = [J_theta0(i);J_theta1(j)];
    Jvals(i,j) = computeCost(xt,Y,thetaj);
  end
 end
 
 %surface Plot
 Jvals = Jvals';
 figure;
 surf(J_theta0,J_theta1,Jvals)
 xlabel('theta0');
 ylabel('theta1');
 
 %contour Plot
 
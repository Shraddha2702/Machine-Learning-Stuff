function [xnormal mu st] =  normalize(x)
  
  xnormal = x;
  mu = zeros(1, size(x,2));
  st = zeros(1, size(x,2));
  
  m = size(x,2);
  n = size(x,1);
  printf('m %d n %d',m,n);
  
  for i = 1:m
    mu(i) = mean(xnormal(:,i));
    st(i) = std(xnormal(:,i));
   end
   
   for i = 1:m
    for j = 1:n
      xnormal(j,i) = (xnormal(j,i)-mu(i))/st(i);
      %printf('\ni %d j %d mu %d st %d xnormal %d',i,j,mu(i),st(i),xnormal(j,i));
    end
   end
   
end
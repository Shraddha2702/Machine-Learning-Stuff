function gz = sigmoid(z)
  gz = 0;
  gz = 1 ./ (1 + e.^(-z));
end
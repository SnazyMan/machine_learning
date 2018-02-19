function [y] = neuralNetRegPredRELU(W1, W2, b1, b2, testdata)
% estimate new examples
    z1 = testdata*W1 + b1;
    a1 = ReLU(z1);
    y = a1*W2 + b2;

end

function g = ReLU(u)
    % Calculates ReLU of u
   g = max(0, u);
end

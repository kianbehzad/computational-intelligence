
[a, b] = sigmatrix([1 2 3; 1 2 3]);
a
b

function sig = sigmoid(x)
    sig = (2/ (1+exp(-x)) - 1);
end

function sig = dSigmoid(x)
    sig = (1 - sigmoid(x).^2) / 2;
end

function [sumSigmoid, sumDSigmoid] = sigmatrix(a)
    mySize = size(a);
    d = mySize(1)+mySize(2) + 1;
    sumSigmoid = 0;
    sumDSigmoid = 0;
    for i = 1:d
        sumSigmoid = sumSigmoid + sigmoid(a(i));
        sumDSigmoid = sumDSigmoid + dSigmoid(a(i));
    end

end
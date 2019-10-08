clc;
clear;

l = sigmoidMatrix([1 2 3; 4 5 10])
h = dSigmoidMatrix([1 2 3; 4 5 10])



function sig = sigmoid(x)
    sig = (1/ (1+exp(-x)));
end

function sig = sigmoidMatrix(a)
    mySize = size(a);
    sig = zeros(mySize(1), mySize(2));
    d = mySize(1)+mySize(2) + 1;
    for i = 1:d
        sig(i) = sigmoid(a(i));
    end
end

function sig = dSigmoidMatrix(x)
    sig = sigmoidMatrix(x) .* (1 - sigmoidMatrix(x));
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
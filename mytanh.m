clc;
clear;

a = tanhmatrix([1 2 3; 1 2 3]);
a


function res = tanh(x)
    res = (exp(x) - exp(-x)) / (exp(x) + exp(-x));
end

function [sumTanh] = tanhmatrix(a)
    mySize = size(a);
    d = mySize(1)+mySize(2) + 1;
    sumTanh = 0;
    for i = 1:d
        sumTanh = sumTanh + tanh(a(i));
    end

end
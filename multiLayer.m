clear;
clc;


error = 0;
y = [1, 1, -1; 1, 0, -1; 0, 1, -1; 0, 0, -1];
d = [0, 1, 1, 0];
w1 = [.1, .2, .3; .2, .3, .1];
w2 = [.1, .1];
a = [0, 0];
counter = 1;
step = 0;

while true
    a(1) = sigmoid(w1(1,:) * transpose(y(counter,:)));
    a(2) = sigmoid(w1(2,:) * transpose(y(counter,:)));
    o = w2 * a';
    error = error + (o - d(counter)) ^ 2;
    
    if(counter == 4)
        step = step + 1;
        error
        if(error < .001)
            break
        end
        error = 0;
        counter = 0;
    end
    counter = counter + 1;

    F = [dSigmoid(a(1)), 0; 0, dSigmoid(a(2))];
    s2 = -0.01 * 1 * (d(counter) - o);
    s1 = F * transpose(w2) * s2;
    
    w1 = w1 - F * transpose(w2) * s2 * y(counter,:);
    w2 = w2 - s2 * a;
    
end

function sig = sigmoid(x)
    sig = (1/ (1+exp(-x)));
end

function sig = dSigmoid(x)
    sig = sigmoid(x) .* (1 - sigmoid(x));
end



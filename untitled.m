x = zeros(100, 1);
for i = 1:100
   x(i) = i; 
end

young = gaussmf(x, [13, 25]);
old = gaussmf(x, [13, 75]);
cmp = compliment(young);

figure(1)
plot(x, young, x, cmp);
figure(2)
plot(x, snorm(old, young), x, tnorm(old, young));

function [cmp] = compliment(x)
    cmp = 1 - x;
end

function [s] = snorm(x, y)
    s = max(x, y);
end

function [t] = tnorm(x, y)
    t = min(x, y);
end


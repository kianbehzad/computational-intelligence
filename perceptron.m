clc;


w = [0.1, 0.1, 0.5];
y = [1, 1, -1; 1, 0, -1; 0, 1, -1; 0, 0, -1];
d = [1, 0, 0, 0];

k = 1;
p = 1;
e = 0;

while true
    net = w * transpose(y(p, 1:end));
    o = hardlim(net);
    r = d(p) - o;
    e = e + r.*r;
    deltaW = r * y(p, 1:end);
    w = w + deltaW * 0.1;
    p = p+1;
    k = k+1;
    if(p >= 4)
        p = 1;
        if(e == 0)
            break
        end
    e = 0;
    end
end
w
for i = 1:4
    net = w * transpose(y(i, 1:end));
    o = hardlim(net)
end
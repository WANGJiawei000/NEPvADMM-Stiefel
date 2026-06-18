D = 90; n = D;
c = 5; p = c; 
d = D - c;

N = 500;  % inlier number
M = 1500;  % outlier number

S = orth(randn(D,d));
X = S*randn(d,N);
O = randn(D,M);
Xtilde = [X O];
Y = normc(Xtilde);

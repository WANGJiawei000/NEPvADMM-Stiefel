function [X_Rsub,F_Sub,sparsity,time_Rsub,i,succ_flag] = RSG_sPCA(B,option)

tic;
r=option.r;
n=option.n;
mu = option.mu;
maxiter =option.maxiter + 1;
tol = option.tol;
 
X = option.phi_init;

if option.type == 1
    AX = B*X;
else
    AX = B'*(B*X);
end

h=@(X) mu*sum(sum(abs(X)));

f_re_sub = zeros(maxiter,1); 
succ_flag = 0; 
f_re_sub(1) = -0.5*sum(sum(X.*(AX))) + h(X);

for i = 2:maxiter
    
    gx = AX - mu*sign(X) ; %negative Euclidean gradient
    xgx = X'*gx;
    pgx = gx - 0.5*X*(xgx+xgx');   %negative  Riemannian gradient using Euclidean metric
    %pgx = gx;
    %eta = 0.6*0.99^i; 
    eta = 1/i^(3/4);  
    %eta = 1/i;  
    X = X + eta * pgx;    % Riemannian step
    %[q,~] = qr(q);    % retraction
    [U, SIGMA, S] = svd(X'*X);   SIGMA =diag(SIGMA);    X = X*(U*diag(sqrt(1./SIGMA))*S');

    if option.type == 1
        AX = B*X;
    else
        AX = B'*(B*X);
    end

    f_re_sub(i) = -0.5*sum(sum(X.*(AX))) + h(X);
    %if  f_re_sub(i) < option.F_manpg + option.tol
    if abs(f_re_sub(i) - f_re_sub(i-1)) <= 1e-6
        succ_flag = 1;
        break;
    end
end

time_Rsub = toc;

X((abs(X)<=1e-5))=0;
X_Rsub = X;

sparsity= sum(sum(X_Rsub==0))/(n*r);
F_Sub = f_re_sub(i-1);
 
%plot(f_re_sub);
%hold on;
%plot(norm_grad);
fprintf('Rsub:Iter ***  Fval *** CPU  **** sparsity \n');    
print_format = ' %i     %1.5e    %1.2f     %1.2f    \n';
fprintf(1,print_format, i-1, F_Sub, time_Rsub, sparsity);

end


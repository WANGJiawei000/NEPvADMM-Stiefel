clc; close all; clear

%% Problem Generating
D = 50; n = D;
c = 5; p = c; 
d = D - c;
trial_number = 1;

N = 700;  % inlier number
M = 1200;  % outlier number

max_iter = 1000;
max_time = 5.0;  % maximum run in seconds

Time_PSGM = 0; Fval_PSGM = 0;
Time_ManPPA = 0; Fval_ManPPA = 0;
Time_IRLS = 0; Fval_IRLS = 0;

F_val_soc_avg = zeros([1,max_iter]);
F_val_madmm_avg = zeros([1,max_iter]);
F_val_radmm_avg = zeros([1,max_iter]);
F_val_nepv_avg = zeros([1,max_iter]);

cpu_time_soc = zeros([trial_number, max_iter]); cpu_time_soc(1) = eps;
cpu_time_madmm = zeros([trial_number, max_iter]); cpu_time_madmm(1) = eps;
cpu_time_radmm = zeros([trial_number, max_iter]); cpu_time_radmm(1) = eps;
cpu_time_nepv = zeros([trial_number, max_iter]); cpu_time_nepv(1) = eps;

vio_soc_avg = zeros([1, max_iter]);
vio_madmm_avg = zeros([1, max_iter]);
vio_radmm_avg = zeros([1, max_iter]);
vio_nepv_avg = zeros([1, max_iter]);

error_X_nepv = zeros([1, trial_number]);
error_X_radmm = zeros([1, trial_number]);
error_X_madmm = zeros([1, trial_number]);
error_X_soc = zeros([1, trial_number]);
error_X_manppa = zeros([1, trial_number]);
error_X_psgm = zeros([1, trial_number]);

iter0 = max_iter; iter1 = max_iter; iter2 = max_iter; iter3 = max_iter; iter4 = max_iter;

for k = 1:trial_number
    rng('shuffle');
    S = orth(randn(D,d));
    X = S*randn(d,N);
    O = randn(D,M);
    Xtilde = [X O];
    %fprintf('Inlier number: %d, outlier number: %d, tiral: %d\n', N, M, k);
    Y = normc(Xtilde);
    F = @(X) sum(sum(abs(Y'*X)));
    
    % random initialize
    X0 = orth(randn(n, p));

    F_val_soc(1) = F(X0);
    F_val_madmm(1) = F(X0);
    F_val_radmm(1) = F(X0);
    
    F_val_soc_avg(1) = F_val_soc_avg(1) + F(X0);
    F_val_madmm_avg(1) = F_val_madmm_avg(1) + F(X0);
    F_val_radmm_avg(1) = F_val_radmm_avg(1) + F(X0);
    

    %% SOC
    X = X0; W = X0;
    Lambda = zeros(size(X));
    eta = 5e-5; rho = 3e2;

    for iter=2:max_iter
        temp_F = @(X) F(X) + rho / 2 * norm(X - W + Lambda, "fro")^2;
        admm_start = tic;

        % X step, subgradient step
        for i=1:100
            subg = Y*sign(Y.'*X) + rho * (X - W + Lambda);
            % disp(i+ "-th X step for SOC, norm: " + norm(subg, 'fro') + ", X fval: " + temp_F(X));
            if norm(subg, 'fro') < 1e-5
                break;
            end
            X = X - eta * subg;
        end

        % W step: a projection step
        [U,~,V] = svd(X + Lambda);
        W = U*eye(n,p)*V.';

        % Lambda step
        Lambda = Lambda + (X - W);

        elapsed_time = toc(admm_start);

        % Value update
        F_val_soc(iter) = F(W);
        F_val_soc_avg(iter) = F_val_soc_avg(iter) + F(W);
        vio_soc_avg(iter) = vio_soc_avg(iter) + norm(W - X, 'fro')/norm(X,'fro');

        if iter >= 3
            if abs(F_val_soc(iter) - F_val_soc(iter-1)) <= 1e-5
                break
            end
        end

        cpu_time_soc(k,iter) = cpu_time_soc(k,iter) + elapsed_time;
        cpu_time_soc(k,iter+1) = cpu_time_soc(k,iter);

        % fprintf('iter: %d, Lagrangian value: %f, function value:%f\n', iter, L_val(iter), F_val(iter));
    end
    iter0 = min(iter, iter0);
    
    error_X_soc(k) = norm(X - W, 'fro')/norm(W,'fro');


    %% MADMM
    X = X0; W = Y.'*X0;
    Lambda = zeros(size(W));
    eta = 1e-5; rho = 120;

    for iter=2:max_iter
        admm_start = tic;

        % X step: a Riemannian gradient step
        for i=1:100

            gx = rho*Y*(Y.'*X - W + Lambda);

            rgx = proj_stiefel(gx, X);

            if norm(rgx, 'fro') < 1e-5

                break;

            end

            X = retr_stiefel(-eta*rgx, X);

        end

        % W step: a l1 minimization step
        W = wthresh(Y.'*X + Lambda ,'s', 1/rho);

        % Lambda step
        YX = Y.'*X;
        Lambda = Lambda + (YX - W);

        elapsed_time = toc(admm_start);

        % Value update
        F_val_madmm(iter) = F(X);
        F_val_madmm_avg(iter) = F_val_madmm_avg(iter) + F(X);
        vio_madmm_avg(iter) = vio_madmm_avg(iter) + norm(Y.'*X - W, 'fro')/norm(Y.'*X,'fro');

        if iter >= 3
            if abs(F_val_madmm(iter) - F_val_madmm(iter-1)) <= 1e-5
                break
            end
        end

        cpu_time_madmm(k,iter) = cpu_time_madmm(k,iter) + elapsed_time;
        cpu_time_madmm(k,iter+1) = cpu_time_madmm(k,iter);

        % fprintf('iter: %d, Lagrangian value: %f, function value:%f\n', iter, L_val(iter), F_val(iter));
    end
    iter1 = min(iter, iter1);

    error_X_madmm(k) = norm(YX - W, 'fro')/norm(YX,'fro');
    

    %% RADMM
    X = X0; W = Y.'*X0; Z = W;
    Lambda = zeros(size(W));
    eta = 4e-4; rho = 50; gamma = 1e-9;
    
    for iter=2:max_iter
        admm_start = tic;

        % X step: a Riemannian gradient step
        for i=1:1
            gx = Y*Lambda + rho*Y*(Y.'*X - Z);
            rgx = proj_stiefel(gx, X);
            
            X = retr_stiefel(-eta*rgx, X);
        end

        % Z step (also update W)
        W = wthresh(Y.' * X + Lambda/rho,'s', (1 + rho * gamma)/rho);
        Z = (W/gamma + Lambda + rho * Y.' * X) / (1/gamma + rho);

        % Lambda step
        Lambda = Lambda + rho * (Y.'*X - Z);

        elapsed_time = toc(admm_start);

        % Value update
        F_val_radmm(iter) = F(X);
        F_val_radmm_avg(iter) = F_val_radmm_avg(iter) + F(X);
        vio_radmm_avg(iter) = vio_radmm_avg(iter) + norm(Y.'*X - W, 'fro')/norm(Y.'*X,'fro');
        if iter >= 3
            if abs(F_val_radmm(iter) - F_val_radmm(iter-1)) <= 1e-6
                break
            end
        end

        cpu_time_radmm(k,iter) = cpu_time_radmm(k,iter) + elapsed_time;
        cpu_time_radmm(k,iter+1) = cpu_time_radmm(k,iter);

        % fprintf('iter: %d, Lagrangian value: %f, function value:%f\n', iter, L_val(iter), F_val(iter));
    end
    iter2 = min(iter, iter2);

    error_X_radmm(k) = norm(Y.'*X - W, 'fro')/norm(Y.'*X, 'fro');


    %% NEPvADMM
    X_nepv = X0; W_nepv = Y' * X0; Z_nepv = W_nepv;
    Lambda_nepv = zeros(size(W_nepv));    
    beta_nepv = 40; gamma_nepv = 2e-8; 

    H_hat = - beta_nepv * (Y * Y');

    D_nepv = Y * (beta_nepv * Z_nepv - Lambda_nepv);

    DX = D_nepv * X_nepv';

    F_val_nepv(1) = F(X_nepv);
    F_val_nepv_avg(1) = F_val_nepv_avg(1) + F_val_nepv(1);

    for iter = 2 : max_iter

        nepv_start = tic;

        % X step: NEPv step
        E = DX + DX' + H_hat;

        [X_nepv_hat, ~] = eigs(E, p, 'largestreal');

        [U_nepv_hat, ~, V_nepv_hat] = svd(X_nepv_hat' * D_nepv, "econ");

        X_nepv = X_nepv_hat * (U_nepv_hat * V_nepv_hat');

        % Y step: PG
        YX_nepv = Y' * X_nepv;

        XL_nepv = YX_nepv + Lambda_nepv / beta_nepv;

        W_nepv = wthresh(XL_nepv, 's', (1 + beta_nepv * gamma_nepv) / beta_nepv);

        Z_nepv = (W_nepv / gamma_nepv + beta_nepv * XL_nepv) / (1/gamma_nepv + beta_nepv);

        % Lambda step:
        Lambda_nepv = Lambda_nepv + beta_nepv*(YX_nepv - Z_nepv);

        D_nepv = Y * (beta_nepv * Z_nepv - Lambda_nepv);

        DX = D_nepv*X_nepv';

        elapsed_time_nepv = toc(nepv_start);

        % Value update
        F_val_nepv(iter) = F(X_nepv);
        F_val_nepv_avg(iter) = F_val_nepv_avg(iter) + F_val_nepv(iter);

        vio_nepv_avg(iter) = vio_nepv_avg(iter) + norm(Y.'*X_nepv - W_nepv, 'fro')/norm(Y.'*X_nepv,'fro');

        if iter >= 3
            if abs((F_val_nepv(iter) - F_val_nepv(iter-1))/F_val_nepv(iter-1)) <= 1e-6
                break
            end
        end

        cpu_time_nepv(k,iter) = cpu_time_nepv(k,iter) + elapsed_time_nepv;
        cpu_time_nepv(k,iter+1) = cpu_time_nepv(k,iter);
    end
    iter3 = min(iter, iter3);

    error_X_nepv(k) = norm(Y.' *X_nepv - W_nepv,'fro')/norm(Y.' *X_nepv,'fro');

   
    %% PSGM
    mu_min = 1e-6; 
    maxiter = 1000;
    [~, B_PSGM,angle_PSGM,time_PSGM,fval_PSGM] = DPCP_PSGM_optim(Y,c,mu_min,maxiter,S, max_time);

    error_X_psgm(k) = norm(B_PSGM.'*B_PSGM - eye(p), 'fro');


    %% ManPPA
    option.maxiter = 1000;
    option.tol = 1e-6;
    option.stepsize = 0.05;  
    option.print_inner = 'off';
    option.exact = 0;
    option.c = c;
    option.S = S;
    option.max_time = max_time;
    [B_ManPPA,~,angle_ManPPA,time_ManPPA, fval_ManPPA] = manppa_DPCP(Y', option);

    fprintf('-------ManPPA principal angle: %e\n', abs(asin(norm(B_ManPPA'*S))));

    error_X_manppa(k) = norm(B_ManPPA.'*B_ManPPA - eye(p), 'fro');   
end

avg = trial_number;

Time_PSGM = Time_PSGM + time_PSGM(end); Fval_PSGM = Fval_PSGM + fval_PSGM(end);
Time_ManPPA = Time_ManPPA + time_ManPPA(end); Fval_ManPPA = Fval_ManPPA + fval_ManPPA(end);

F_val_soc_avg = (F_val_soc_avg/avg);
F_val_madmm_avg = (F_val_madmm_avg/avg);
F_val_radmm_avg = (F_val_radmm_avg/avg);
F_val_nepv_avg = (F_val_nepv_avg/avg);


vio_soc_avg = (vio_soc_avg/avg);
vio_madmm_avg = (vio_madmm_avg/avg);
vio_radmm_avg = (vio_radmm_avg/avg);
vio_nepv_avg = (vio_nepv_avg/avg);

cpu_time_soc = sum(cpu_time_soc,1)/avg;
cpu_time_madmm = sum(cpu_time_madmm,1)/avg;
cpu_time_radmm = sum(cpu_time_radmm,1)/avg;
cpu_time_nepv = sum(cpu_time_nepv,1)/avg;

av_error_nepv = sum(error_X_nepv)/avg;
av_error_radmm = sum(error_X_radmm)/avg;
av_error_madmm = sum(error_X_madmm)/avg;
av_error_soc = sum(error_X_soc)/avg;
av_error_manppa = sum(error_X_manppa)/avg;
av_error_psgm = sum(error_X_psgm)/avg;


disp("error of the first constraint of SOC, MADMM, RADMM and NEPvADMM: ")

disp([vio_soc_avg(iter0 - 1), vio_madmm_avg(iter1 - 1), vio_radmm_avg(iter2 - 1), vio_nepv_avg(iter3 - 1)])

disp("error of the manifold constraint of SOC, MADMM, RADMM, ManPPA, PSGM and NEPvADMM: ")
            
disp([av_error_soc, av_error_madmm, av_error_radmm, av_error_manppa, av_error_psgm, av_error_nepv])

disp("CPU time for SOC, MADMM, RADMM, ManPPA, PSGM and NEPvADMM: ")

disp([cpu_time_soc(iter0 - 1), cpu_time_madmm(iter1 - 1), cpu_time_radmm(iter2 - 1), Time_ManPPA, Time_PSGM, cpu_time_nepv(iter3 - 1)]);

disp("function value for output SOC, MADMM, RADMM, ManPPA, PSGM and NEPvADMM: ")

disp([F_val_soc_avg(iter0 - 1), F_val_madmm_avg(iter1 - 1), F_val_radmm_avg(iter2 - 1), Fval_ManPPA, Fval_PSGM, F_val_nepv_avg(iter3 - 1)]);

%% Plots
%%
plotStyle = {'ro','bo','g-','k:','c-','r:','g:'};
figure0 = figure(1);
clf
plot(F_val_soc_avg(1:iter0), LineWidth = 2); hold on;
plot(F_val_madmm_avg(1:iter1), LineWidth = 2); hold on;
plot(F_val_radmm_avg(1:iter2) ,LineWidth = 2); hold on;
plot(F_val_nepv_avg(1:iter3) ,LineWidth = 2); hold on;
xlabel("Iterations", 'FontSize', 18);
ylabel("Objective value", 'FontSize', 18);
legend('SOC', 'MADMM','RADMM','NEPvADMM', 'FontSize', 11);
legend('Location','best');
filename = "dpcp_soc_madmm_n_" + n + "_p_" + p + "_fval.pdf";
saveas(figure0, filename);
% figure0.show()


figure1 = figure(2);
clf
semilogy(cpu_time_soc(1:iter0), F_val_soc_avg(1:iter0), LineWidth = 2); hold on;
semilogy(cpu_time_madmm(1:iter1), F_val_madmm_avg(1:iter1), LineWidth = 2); hold on;
semilogy(cpu_time_radmm(1:iter2), F_val_radmm_avg(1:iter2) ,LineWidth = 2); hold on;
semilogy(cpu_time_nepv(1:iter3), F_val_nepv_avg(1:iter3) ,LineWidth = 2); hold on;
semilogy(time_ManPPA(end)+eps, fval_ManPPA(end), plotStyle{1},'linewidth',2); hold on;
semilogy(time_PSGM(end)+eps, fval_PSGM(end), plotStyle{2},'linewidth',2); hold on; 
xlabel("CPU time", 'FontSize', 18);
ylabel("Objective value", 'FontSize', 18);
legend('SOC', 'MADMM', 'RADMM', 'NEPvADMM', 'ManPPA', 'PSGM', 'FontSize', 11);
legend('Location','best');
filename = "dpcp_soc_madmm_cpu_time_n_" + n + "_p_" + p + "_fval.pdf";
saveas(figure1, filename);
% figure1.show()

clc; close all; clear


M_out_set = 800 : 50 : 1500;
N_in = 500;

M_out_num = length(M_out_set);

test_number = 1;

index = 1;

max_iter = 1000;
max_time = 5.0;  % maximum run in seconds

iter0 = max_iter; iter1 = max_iter; iter2 = max_iter; iter3 = max_iter; iter4 = max_iter;

F_val_soc_avg = zeros([test_number, M_out_num]);
F_val_madmm_avg = zeros([test_number, M_out_num]);
F_val_radmm_avg = zeros([test_number, M_out_num]);
F_val_nepv_avg = zeros([test_number, M_out_num]);

cpu_time_soc = zeros([test_number, M_out_num]); 
cpu_time_madmm = zeros([test_number, M_out_num]); 
cpu_time_radmm = zeros([test_number, M_out_num]); 
cpu_time_nepv = zeros([test_number, M_out_num]); 

error_X_nepv = zeros([test_number, M_out_num]);
error_X_radmm = zeros([test_number, M_out_num]);
error_X_madmm = zeros([test_number, M_out_num]);
error_X_soc = zeros([test_number, M_out_num]);
error_X_manppa = zeros([test_number, M_out_num]);
error_X_psgm = zeros([test_number, M_out_num]);

iter_nepv = zeros([test_number, M_out_num]);
iter_radmm = zeros([test_number, M_out_num]);
iter_madmm = zeros([test_number, M_out_num]);
iter_soc = zeros([test_number, M_out_num]);

Time_PSGM = zeros([test_number, M_out_num]); Fval_PSGM = zeros([test_number, M_out_num]);
Time_ManPPA = zeros([test_number, M_out_num]); Fval_ManPPA = zeros([test_number, M_out_num]);


for id_n = 1 : size(M_out_set, 2)

    M_out = M_out_set(id_n);

    for test_random = 1 : test_number  %times average.
      
        %% Problem Generating
        D = 90; n = D;
        c = 5; p = c;
        d = D - c;
                
        
        cpu_time_soc(test_random, id_n) = eps;
        cpu_time_madmm(test_random, id_n) = eps;
        cpu_time_radmm(test_random, id_n) = eps;
        cpu_time_nepv(test_random, id_n) = eps;
                        
        
        rng('shuffle');

        S = orth(randn(D, d));
        X = S*randn(d, N_in);
        O = randn(D, M_out);
        Xtilde = [X O];

        Y = normc(Xtilde);

        F = @(X) sum(sum(abs(Y' * X)));
        
        % random initialize
        X0 = orth(randn(n, p));
    
        FX0 = F(X0);

        F_val_soc(1) = FX0;
        F_val_madmm(1) = FX0;
        F_val_radmm(1) = FX0;
        
        F_val_soc_avg(test_random, id_n) = F_val_soc_avg(test_random, id_n) + FX0;
        F_val_madmm_avg(test_random, id_n) = F_val_madmm_avg(test_random, id_n) + FX0;
        F_val_radmm_avg(test_random, id_n) = F_val_radmm_avg(test_random, id_n) + FX0;
        
    
        %% SOC
        X_soc = X0; W_soc = X0;
        Lambda_soc = zeros(size(X_soc));

        eta_soc = 5e-5; rho_soc = 3e2;
    
        for iter = 2 : max_iter

            admm_start_soc = tic;
    
            % X step, subgradient step
            for i = 1 : 100

                subg = Y * sign(Y' * X_soc) + rho_soc * (X_soc - W_soc + Lambda_soc);

                if norm(subg, 'fro') < 1e-5

                    break;

                end

                X_soc = X_soc - eta_soc * subg;

            end
    
            % W step: a projection step
            [U_soc, ~, V_soc] = svd(X_soc + Lambda_soc);

            W_soc = U_soc * eye(n, p) * V_soc';
    
            % Lambda step
            Lambda_soc = Lambda_soc + (X_soc - W_soc);
    
            elapsed_time_soc = toc(admm_start_soc);
    
            % Value update
            F_val_soc(iter) = F(X_soc);
    
            if iter >= 3

                if abs(F_val_soc(iter) - F_val_soc(iter - 1)) <= 1e-5

                    break

                end

            end
    
            cpu_time_soc(test_random, id_n) = cpu_time_soc(test_random, id_n) + elapsed_time_soc;
    
        end

        iter_soc(test_random, id_n) = min(iter, iter0);

        F_val_soc_avg(test_random, id_n) = F_val_soc(iter); 
        
        error_X_soc(test_random, id_n) = norm(X_soc - W_soc, 'fro') / norm(W_soc, 'fro');
    
    
        %% MADMM
        X_madmm = X0; W_madmm = Y' * X0;
        Lambda_madmm = zeros(size(W_madmm));

        eta_madmm = 2e-5; rho_madmm = 120;
    
        for iter = 2 : max_iter

            admm_start_madmm = tic;
    
            % X step: a Riemannian gradient step
            for i = 1 : 100

                gx_madmm = rho_madmm * Y * (Y' * X_madmm - W_madmm + Lambda_madmm);

                rgx_madmm = proj_stiefel(gx_madmm, X_madmm);

                if norm(rgx_madmm, 'fro') < 1e-5

                    break;

                end

                X_madmm = retr_stiefel(- eta_madmm * rgx_madmm, X_madmm);

            end
    
            % W step: a l1 minimization step
            W_madmm = wthresh(Y' * X_madmm + Lambda_madmm ,'s', 1 / rho_madmm);
    
            % Lambda step
            YX_madmm = Y' * X_madmm;

            Lambda_madmm = Lambda_madmm + (YX_madmm - W_madmm);
    
            elapsed_time_madmm = toc(admm_start_madmm);
    
            % Value update
            F_val_madmm(iter) = F(X_madmm);

            %vio_madmm_avg(iter) = vio_madmm_avg(iter) + norm(Y.'*X - W, 'fro')/norm(Y.'*X,'fro');
    
            if iter >= 3

                if abs(F_val_madmm(iter) - F_val_madmm(iter - 1)) <= 1e-5

                    break

                end

            end
    
            cpu_time_madmm(test_random, id_n) = cpu_time_madmm(test_random, id_n) + elapsed_time_madmm;
    
        end

        iter_madmm(test_random, id_n) = min(iter, iter1);

        F_val_madmm_avg(test_random, id_n) = F_val_madmm(iter);
    
        error_X_madmm(test_random, id_n) = norm(YX_madmm - W_madmm, 'fro') / norm(YX_madmm, 'fro');
        
    
        %% RADMM
        X_radmm = X0; W_radmm = Y' * X0; Z_radmm = W_radmm;
        Lambda_radmm = zeros(size(W_radmm));

        eta_radmm = 4e-4; rho_radmm = 50; gamma_radmm = 1e-9;
        
        for iter = 2 : max_iter

            admm_start_radmm = tic;
    
            % X step: a Riemannian gradient step
            for i = 1 : 1

                gx_radmm = Y * Lambda_radmm + rho_radmm * Y * (Y' * X_radmm - Z_radmm);

                rgx_radmm = proj_stiefel(gx_radmm, X_radmm);

                X_radmm = retr_stiefel(- eta_radmm * rgx_radmm, X_radmm);

            end
    
            % Z step (also update W)
            W_radmm = wthresh(Y' * X_radmm + Lambda_radmm / rho_radmm, 's', (1 + rho_radmm * gamma_radmm)/rho_radmm);

            Z_radmm = (W_radmm / gamma_radmm + Lambda_radmm + rho_radmm * Y' * X_radmm) / (1 / gamma_radmm + rho_radmm);
    
            % Lambda step
            Lambda_radmm = Lambda_radmm + rho_radmm * (Y' * X_radmm - Z_radmm);
    
            elapsed_time_radmm = toc(admm_start_radmm);
    
            % Value update
            F_val_radmm(iter) = F(X_radmm);

            %vio_radmm_avg(iter) = vio_radmm_avg(iter) + norm(Y.'*X - W, 'fro')/norm(Y.'*X,'fro');

            if iter >= 3

                if abs(F_val_radmm(iter) - F_val_radmm(iter - 1)) <= 1e-6

                    break

                end

            end
    
            cpu_time_radmm(test_random, id_n) = cpu_time_radmm(test_random, id_n) + elapsed_time_radmm;
    
        end

        iter_radmm(test_random, id_n) = min(iter, iter2);

        F_val_radmm_avg(test_random, id_n) = F_val_radmm(iter);
    
        error_X_radmm(test_random, id_n) = norm(Y' * X_radmm - W_radmm, 'fro') / norm(Y' * X_radmm, 'fro');
    
    
        %% NEPvADMM
        X_nepv = X0; W_nepv = Y' * X0; Z_nepv = W_nepv;
        Lambda_nepv = zeros(size(W_nepv));  

        beta_nepv = 40; gamma_nepv = 2e-8; 
    
        H_hat = - beta_nepv * (Y * Y');

        w_cons = (1 + beta_nepv * gamma_nepv) / beta_nepv;

        z_cons = 1/gamma_nepv + beta_nepv;
    
        D_nepv = Y * (beta_nepv * Z_nepv - Lambda_nepv);
    
        DX = D_nepv * X_nepv';
        
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
    
            W_nepv = wthresh(XL_nepv, 's', w_cons);
    
            Z_nepv = (W_nepv / gamma_nepv + beta_nepv * XL_nepv) / z_cons;
    
            % Lambda step:
            Lambda_nepv = Lambda_nepv + beta_nepv * (YX_nepv - Z_nepv);
    
            D_nepv = Y * (beta_nepv * Z_nepv - Lambda_nepv);
    
            DX = D_nepv * X_nepv';
    
            elapsed_time_nepv = toc(nepv_start);
    
            % Value update
            F_val_nepv(iter) = F(X_nepv);
    
            %vio_nepv_avg(iter) = vio_nepv_avg(iter) + norm(Y.'*X_nepv - W_nepv, 'fro')/norm(Y.'*X_nepv,'fro');
    
            if iter >= 3

                if abs((F_val_nepv(iter) - F_val_nepv(iter-1)) / F_val_nepv(iter-1)) <= 1e-6

                    break

                end

            end
    
            cpu_time_nepv(test_random, id_n) = cpu_time_nepv(test_random, id_n) + elapsed_time_nepv;

        end

        iter_nepv(test_random, id_n) = min(iter, iter3); 

        F_val_nepv_avg(test_random, id_n) = F_val_nepv(iter);

        error_X_nepv(test_random, id_n) = norm(Y' * X_nepv - W_nepv, 'fro') / norm(Y' * X_nepv, 'fro');
    
       
        %% PSGM
        mu_min = 1e-6; 
        maxiter = 1000;

        [~, B_PSGM, angle_PSGM, time_PSGM, fval_PSGM] = DPCP_PSGM_optim(Y, c, mu_min, maxiter, S, max_time);
        
    
        %% ManPPA
        option.maxiter = 1000; 
        option.tol = 1e-6;
        option.stepsize = 0.05;  
        option.print_inner = 'off';
        option.exact = 0;
        option.c = c;
        option.S = S;
        option.max_time = max_time;

        [B_ManPPA, ~, angle_ManPPA, time_ManPPA, fval_ManPPA] = manppa_DPCP(Y', option);
        
        
        Time_PSGM(test_random, id_n) = Time_PSGM(test_random, id_n) + time_PSGM(end); 

        Fval_PSGM(test_random, id_n) = Fval_PSGM(test_random, id_n) + fval_PSGM(end);

        Time_ManPPA(test_random, id_n) = Time_ManPPA(test_random, id_n) + time_ManPPA(end); 
        
        Fval_ManPPA(test_random, id_n) = Fval_ManPPA(test_random, id_n) + fval_ManPPA(end);
                      
    end 
       
end


Fval_soc = sum(F_val_soc_avg, 1) / test_number;
Fval_madmm = sum(F_val_madmm_avg, 1) / test_number;
Fval_radmm = sum(F_val_radmm_avg, 1) / test_number;
Fval_nepv = sum(F_val_nepv_avg, 1) / test_number;
Fval_ps = sum(Fval_PSGM, 1) / test_number;
Fval_man = sum(Fval_ManPPA, 1) / test_number;


t_soc = sum(cpu_time_soc, 1) / test_number;
t_madmm = sum(cpu_time_madmm, 1) / test_number;
t_radmm = sum(cpu_time_radmm, 1) / test_number;
t_nepv = sum(cpu_time_nepv, 1) / test_number;
t_psgm = sum(Time_PSGM, 1) / test_number;
t_manppa = sum(Time_ManPPA, 1) / test_number;


it_soc = sum(iter_soc, 1) / test_number;
it_madmm = sum(iter_madmm, 1) / test_number;
it_radmm = sum(iter_radmm, 1) / test_number;
it_nepv = sum(iter_nepv, 1) / test_number;

er_soc = sum(error_X_soc, 1) / test_number;
er_madmm = sum(error_X_madmm, 1) / test_number;
er_radmm = sum(error_X_radmm, 1) / test_number;
er_nepv = sum(error_X_nepv, 1) / test_number;


%% plot
figure(1);
plot(M_out_set, t_soc, 'r-s', 'MarkerSize', 10, 'linewidth', 1); hold on;
plot(M_out_set, t_madmm, 'k-o', 'MarkerSize', 6, 'linewidth', 1); hold on;
plot(M_out_set, t_radmm, 'b-d', 'MarkerSize', 8, 'linewidth', 1); hold on;
plot(M_out_set, t_nepv, 'c-^', 'MarkerSize', 20, 'linewidth', 1); hold on;
plot(M_out_set, t_psgm, 'm-*', 'MarkerSize', 20, 'linewidth', 1); hold on;
plot(M_out_set, t_manppa, 'g-.', 'MarkerSize', 20, 'linewidth', 1);
xlabel('p_{2}', 'FontSize', 18);   ylabel('CPU', 'FontSize', 18);
legend('SOC', 'MADMM', 'RADMM', 'NEPvADMM', 'PSGM', 'ManPPA', 'FontSize', 10);

ylim([-2  25]); 


figure(2)
semilogy(M_out_set, Fval_soc, 'r-s','MarkerSize',10,'linewidth',1); hold on;
semilogy(M_out_set, Fval_madmm, 'k-o','MarkerSize',6,'linewidth',1); hold on;
semilogy(M_out_set, Fval_radmm, 'b-d','MarkerSize',8,'linewidth',1); hold on;
semilogy(M_out_set, Fval_nepv, 'c-^','MarkerSize',20,'linewidth',1.5); hold on;
semilogy(M_out_set, Fval_ps, 'm-*','MarkerSize',20,'linewidth',1.5); hold on;
semilogy(M_out_set, Fval_man, 'g-.','MarkerSize',20,'linewidth',1.5);
legend('SOC', 'MADMM', 'RADMM', 'NEPvADMM', 'PSGM', 'ManPPA', 'FontSize', 10);
xlabel('p_{2}', 'FontSize', 18);   ylabel('objective fucntion value', 'FontSize', 18);


figure(3)
plot(M_out_set, it_soc, 'r-s','MarkerSize',10,'linewidth',1); hold on;
plot(M_out_set, it_madmm, 'k-o','MarkerSize',6,'linewidth',1); hold on;
plot(M_out_set, it_radmm, 'c-^','MarkerSize',20,'linewidth',1.5); hold on;
plot(M_out_set, it_nepv, 'm-*','MarkerSize',20,'linewidth',1.5); hold on;
xlabel('p_{2}', 'FontSize', 18);   ylabel('iter', 'FontSize', 18);
legend('SOC', 'MADMM', 'RADMM', 'NEPvADMM', 'FontSize', 10);

ylim([100  1100]); 


figure(4)
plot(M_out_set, er_soc, 'r-s','MarkerSize',10,'linewidth',1); hold on;
plot(M_out_set, er_madmm, 'k-o','MarkerSize',6,'linewidth',1); hold on;
plot(M_out_set, er_radmm, 'c-^','MarkerSize',20,'linewidth',1.5); hold on;
plot(M_out_set, er_nepv, 'm-*','MarkerSize',20,'linewidth',1.5); hold on;
xlabel('p_{2}', 'FontSize', 18);   ylabel('error', 'FontSize', 18);
legend('SOC', 'MADMM', 'RADMM', 'NEPvADMM', 'FontSize', 10);


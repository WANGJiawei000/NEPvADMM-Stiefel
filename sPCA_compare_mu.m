% Comparsion between NEPvADMM with ManPG, Subgradient, MADMM and RADMM Method on sPCA
clc; close all; clear

addpath misc
addpath SSN_subproblem

n_set=[100; 150; 200; 250; 300;];
r_set = [2; 5; 10; 15; 30];
mu_set = [0.5; 0.7; 0.9; 1;];
test_number = 50;

index = 1;

for id_mu = 1:size(mu_set,1)

    mu = mu_set(id_mu);
    fid =1;
    
    for id_r = 2
        for id_n = 5          

            r = r_set(id_r);
            n = n_set(id_n);
            
            succ_no_manpg = 0; succ_no_SOC = 0; succ_no_sub = 0; succ_no_madmm = 0; succ_no_radmm = 0; succ_no_nepv = 0;
            fail_no_SOC = 0; fail_no_sub = 0; fail_no_madmm = 0;  fail_no_radmm = 0;  fail_no_nepv = 0;

            for test_random = 1:test_number  %times average.
                fprintf(fid,'==============================================================================================\n');
                
                rng('shuffle');

                M = randn(n,n); M = (M+M')/2; [U1,~] = eig(M);
                v = rand(n,1) + 1e-6; B = U1*diag(v)*U1';
                type = 1;


                fprintf(fid,'- n -- r -- mu --------\n');
                fprintf(fid,'%4d %3d %3.3f \n',n,r,mu);
                fprintf(fid,'----------------------------------------------------------------------------------\n');
                
                rng('shuffle');


                phi_init = randn(n, r);
                phi_init = orth(phi_init);
                %[phi_init,~] = svd(randn(n,r),0);  % random intialization

                
                %%%%% ManPG parameter
                option_manpg.adap = 0;    option_manpg.type =type;
                option_manpg.phi_init = phi_init; option_manpg.maxiter = 2000;  option_manpg.tol = 1e-6;
                option_manpg.r = r;    option_manpg.n = n;  option_manpg.mu = mu;
                option_manpg.inner_iter = 100;

                [X_manpg, F_manpg(test_random), sparsity_manpg(test_random), time_manpg(test_random),...
                    manpg_error(test_random), maxit_att_manpg(test_random), succ_flag_manpg, lins(test_random),in_av(test_random)]= ManPG_sPCA(B,option_manpg);

                if succ_flag_manpg == 1
                    succ_no_manpg = succ_no_manpg + 1;
                end  

                if succ_flag_manpg == 0
                    fail_no_manpg = fail_no_manpg + 1;
                end                

                %%%%%% SOC parameter
                option_soc.phi_init = phi_init; option_soc.maxiter = 2000;  option_soc.tol = 1e-6;
                option_soc.r = r;    option_soc.n = n;  option_soc.mu = mu;
                option_soc.type = type;

                [X_Soc, F_soc(test_random),sparsity_soc(test_random),time_soc(test_random),...
                    soc_error_XPQ(test_random),maxit_att_soc(test_random),succ_flag_SOC]= SOC_sPCA(B,option_soc);

                if succ_flag_SOC == 1
                    succ_no_SOC = succ_no_SOC + 1;
                end

                if succ_flag_SOC == 0
                    fail_no_SOC = fail_no_SOC + 1;
                end                
              
                %%%%%% Riemannian subgradient parameter
                option_Rsub.phi_init = phi_init; option_Rsub.maxiter = 1e3;  option_Rsub.tol = 5e-3;
                option_Rsub.r = r;    option_Rsub.n = n;  option_Rsub.mu = mu;  option_Rsub.type = type;
                
                [X_Rsub, F_Rsub(test_random),sparsity_Rsub(test_random),time_Rsub(test_random),...
                    maxit_att_Rsub(test_random),succ_flag_sub]= RSG_sPCA(B,option_Rsub);

                if succ_flag_sub == 1
                    succ_no_sub = succ_no_sub + 1;
                end

                if succ_flag_sub == 0
                    fail_no_sub = fail_no_sub + 1;
                end                

                %%%%%% MADMM parameter
                option_madmm.phi_init = phi_init; option_madmm.maxiter = 2000; option_madmm.tol = 1e-6;
                option_madmm.r = r;    option_madmm.n = n;  option_madmm.mu = mu;
                option_madmm.type = type;

                [X_madmm, F_madmm(test_random), sparsity_madmm(test_random), time_madmm(test_random),...
                    error_madmm(test_random), maxit_att_madmm(test_random), succ_flag_madmm]= MADMM_sPCA(B,option_madmm);

                if succ_flag_madmm == 1
                    succ_no_madmm = succ_no_madmm + 1;
                end

                if succ_flag_madmm == 0
                    fail_no_madmm = fail_no_madmm + 1;
                end                
                
                %%%%%% RADMM parameter
                option_radmm.phi_init = phi_init; option_radmm.maxiter = 2000;  option_radmm.tol = 1e-6;
                option_radmm.r = r;    option_radmm.n = n;  option_radmm.mu = mu;
                option_radmm.type = type;

                [X_radmm, F_radmm(test_random), sparsity_radmm(test_random), time_radmm(test_random),...
                    error_radmm(test_random), maxit_att_radmm(test_random), succ_flag_radmm]= RADMM_sPCA(B,option_radmm);

                if succ_flag_radmm == 1
                    succ_no_radmm = succ_no_radmm + 1;
                end

                if succ_flag_radmm == 0
                    fail_no_radmm = fail_no_radmm + 1;
                end                

                %%%%%% NEPvADMM parameter
                option_nepv.phi_init = phi_init; option_nepv.maxiter = 2000; option_nepv.tol = 1e-6;
                option_nepv.r = r;   option_nepv.n = n;  option_nepv.mu = mu;
                option_nepv.type = type;

                [X_nepv, F_nepv(test_random), sparsity_nepv(test_random), time_nepv(test_random),...
                    error_nepv(test_random), maxit_att_nepv(test_random), succ_flag_nepv] = NEPvADMM_sPCA(B,option_nepv);

                if succ_flag_nepv == 1
                    succ_no_nepv = succ_no_nepv + 1;
                end

                if succ_flag_nepv == 0
                    fail_no_nepv = fail_no_nepv + 1;
                end


                if succ_flag_manpg == 1
                    F_best(test_random) =  F_manpg(test_random);
                end

                if succ_flag_SOC == 1
                    F_best(test_random) =  min( F_best(test_random), F_soc(test_random));
                end

                if succ_flag_madmm == 1
                    F_best(test_random) =  min( F_best(test_random), F_madmm(test_random));
                end
                
                if succ_flag_radmm == 1
                    F_best(test_random) =  min( F_best(test_random), F_radmm(test_random));
                end

                if succ_flag_nepv == 1
                    F_best(test_random) =  min( F_best(test_random), F_nepv(test_random));
                end

                if succ_flag_sub == 1
                    F_best(test_random) =  min( F_best(test_random), F_Rsub(test_random));
                end
            end
            
        end
        
    end
    

    index = index + 1;
    
    iter.manpg(id_mu) =  sum(maxit_att_manpg)/succ_no_manpg;
    iter.soc(id_mu) =  sum(maxit_att_soc)/succ_no_SOC;
    iter.Rsub(id_mu) =  sum(maxit_att_Rsub)/succ_no_sub;
    iter.madmm(id_mu) =  sum(maxit_att_madmm)/succ_no_madmm;
    iter.radmm(id_mu) =  sum(maxit_att_radmm)/succ_no_radmm;
    iter.nepv(id_mu) =  sum(maxit_att_nepv)/succ_no_nepv;
    
    time.manpg(id_mu) =  sum(time_manpg)/succ_no_manpg;
    time.soc(id_mu) =  sum(time_soc)/succ_no_SOC;
    time.Rsub(id_mu) =  sum(time_Rsub)/succ_no_sub;
    time.madmm(id_mu) =  sum(time_madmm)/succ_no_madmm;
    time.radmm(id_mu) =  sum(time_radmm)/succ_no_radmm;
    time.nepv(id_mu) =  sum(time_nepv)/succ_no_nepv;
    
    Fval.manpg(id_mu) =  sum(F_manpg)/succ_no_manpg;
    Fval.soc(id_mu) =  sum(F_soc)/succ_no_SOC;
    Fval.Rsub(id_mu) =  sum(F_Rsub)/succ_no_sub;
    Fval.madmm(id_mu) =  sum(F_madmm)/succ_no_madmm;
    Fval.radmm(id_mu) =  sum(F_radmm)/succ_no_radmm;
    Fval.nepv(id_mu) =  sum(F_nepv)/succ_no_nepv;
    
    Sp.manpg(id_mu) =  sum(sparsity_manpg)/succ_no_manpg;
    Sp.soc(id_mu) =  sum(sparsity_soc)/succ_no_SOC;
    Sp.Rsub(id_mu) =  sum(sparsity_Rsub)/succ_no_sub;
    Sp.madmm(id_mu) =  sum(sparsity_madmm)/succ_no_madmm;
    Sp.radmm(id_mu) =  sum(sparsity_radmm)/succ_no_radmm;
    Sp.nepv(id_mu) =  sum(sparsity_nepv)/succ_no_nepv;

    fprintf(fid,'==============================================================================================\n');
    
    fprintf(fid,' Alg ****        Iter *****  Fval *** sparsity ** cpu *** Error ***\n');
    
    print_format =  'ManPG:      %1.3e  %1.5e    %1.2f      %3.2f    %1.3e\n';
    fprintf(fid,print_format, iter.manpg(id_mu), Fval.manpg(id_mu), Sp.manpg(id_mu),time.manpg(id_mu), mean(manpg_error));
    print_format =  'SOC:        %1.3e  %1.5e    %1.2f      %3.2f    %1.3e\n';
    fprintf(fid,print_format,iter.soc(id_mu) , Fval.soc(id_mu), Sp.soc(id_mu) ,time.soc(id_mu),mean(soc_error_XPQ));
    print_format =  'Rsub:       %1.3e  %1.5e    %1.2f      %3.2f  \n';
    fprintf(fid,print_format,iter.Rsub(id_mu) ,  Fval.Rsub(id_mu) ,Sp.Rsub(id_mu),time.Rsub(id_mu));
    print_format =  'MADMM:        %1.3e  %1.5e    %1.2f      %3.2f    %1.3e\n';
    fprintf(fid,print_format,iter.madmm(id_mu) , Fval.madmm(id_mu), Sp.madmm(id_mu) ,time.madmm(id_mu),mean(error_madmm));
    print_format =  'RADMM:        %1.3e  %1.5e    %1.2f      %3.2f    %1.3e\n';
    fprintf(fid,print_format,iter.radmm(id_mu) , Fval.radmm(id_mu), Sp.radmm(id_mu) ,time.radmm(id_mu),mean(error_radmm));
    print_format =  'NEPvADMM:        %1.3e  %1.5e    %1.2f      %3.2f    %1.3e\n';
    fprintf(fid,print_format,iter.nepv(id_mu) , Fval.nepv(id_mu), Sp.nepv(id_mu) ,time.nepv(id_mu), mean(error_nepv));
end


%% plot
figure(1);
plot(mu_set, time.manpg,'r-s','MarkerSize',10,'linewidth',1); hold on;
plot(mu_set, time.madmm,'k-o','MarkerSize',6,'linewidth',1); hold on;
plot(mu_set, time.soc,'b-d','MarkerSize',8,'linewidth',1); hold on;
plot(mu_set, time.radmm,'c-^','MarkerSize',20,'linewidth',1); hold on;
plot(mu_set, time.nepv,'m-*','MarkerSize',20,'linewidth',1); hold on;
%plot(mu_set, time.Rsub,'g-.','MarkerSize',20,'linewidth',1);
xlabel('\mu', 'FontSize', 18);   ylabel('CPU', 'FontSize', 18);
%title(['comparison on CPU: different dimension',',r=',num2str(r),',\mu=',num2str(mu)]);
legend('ManPG','MADMM','SOC','RADMM','NEPvADMM', 'FontSize', 11);
%filename_pic1 = ['SPCA_CPU_n',  '_' num2str(r) '_' num2str(mu)  '.eps'];
%saveas(gcf,filename_pic1,'epsc')


figure(2)
semilogy(mu_set, Fval.manpg, 'r-s','MarkerSize',10,'linewidth',1); hold on;
semilogy(mu_set, Fval.madmm, 'k-o','MarkerSize',6,'linewidth',1); hold on;
semilogy(mu_set, Fval.soc, 'b-d','MarkerSize',8,'linewidth',1); hold on;
semilogy(mu_set, Fval.radmm, 'c-^','MarkerSize',20,'linewidth',1.5); hold on;
semilogy(mu_set, Fval.nepv, 'm-*','MarkerSize',20,'linewidth',1.5); hold on;
%semilogy(mu_set, Fval.Rsub, 'g-.','MarkerSize',20,'linewidth',1.5);
legend('ManPG','MADMM','SOC','RADMM','NEPvADMM', 'FontSize', 11);
xlabel('\mu', 'FontSize', 18);   ylabel('objective fucntion value', 'FontSize', 18);
%title(['comparison on objective function value: different dimension',',r=',num2str(r),',\mu=',num2str(mu)]);
%filename_pic2 = ['SPCA_Fval_n',  '_' num2str(r) '_' num2str(mu)  '.eps'];
%saveas(gcf,filename_pic2,'epsc')


figure(3)
plot(mu_set, Sp.manpg, 'r-s','MarkerSize',10,'linewidth',1); hold on;
plot(mu_set, Sp.madmm, 'k-o','MarkerSize',6,'linewidth',1); hold on;
plot(mu_set, Sp.soc, 'b-d','MarkerSize',8,'linewidth',1); hold on;
plot(mu_set, Sp.radmm, 'c-^','MarkerSize',20,'linewidth',1.5); hold on;
plot(mu_set, Sp.nepv, 'm-*','MarkerSize',20,'linewidth',1.5); hold on;
%plot(mu_set, Sp.Rsub, 'g-.','MarkerSize',20,'linewidth',1.5);
xlabel('\mu', 'FontSize', 18);   ylabel('sparsity', 'FontSize', 18);
legend('ManPG','MADMM','SOC','RADMM','NEPvADMM', 'FontSize', 11);
%title(['comparison on sparsity: different dimension',',r=',num2str(r),',\mu=',num2str(mu)]);
%filename_pic3 = ['SPCA_Sparsity_n',  '_' num2str(r)  '_' num2str(mu) '.eps'];
%saveas(gcf,filename_pic3,'epsc')


figure(4)
plot(mu_set, iter.manpg, 'r-s','MarkerSize',10,'linewidth',1); hold on;
plot(mu_set, iter.madmm, 'k-o','MarkerSize',6,'linewidth',1); hold on;
plot(mu_set, iter.soc, 'b-d','MarkerSize',8,'linewidth',1); hold on;
plot(mu_set, iter.radmm, 'c-^','MarkerSize',20,'linewidth',1.5); hold on;
plot(mu_set, iter.nepv, 'm-*','MarkerSize',20,'linewidth',1.5); hold on;
%plot(mu_set, iter.Rsub, 'g-.','MarkerSize',20,'linewidth',1.5);
xlabel('\mu', 'FontSize', 18);   ylabel('iter', 'FontSize', 18);
legend('ManPG','MADMM','SOC','RADMM','NEPvADMM', 'FontSize', 11);
%title(['comparison on iter: different dimension',',r=',num2str(r),',\mu=',num2str(mu)]);
%filename_pic4= ['SPCA_iter_n',  '_'  num2str(r)  '_' num2str(mu) '.eps'];
%saveas(gcf,filename_pic4,'epsc')


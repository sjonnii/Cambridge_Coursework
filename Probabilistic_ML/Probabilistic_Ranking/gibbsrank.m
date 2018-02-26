clear;clc
load tennis_data

randn('seed',27); % set the pseudo-random number generator seed

M = size(W,1);            % number of players
N = size(G,1);            % number of games in 2011 season 

pv = 0.5*ones(M,1);           % prior skill variance 

w = zeros(M,1);              % set skills to prior mean
w_plot = zeros(M,1100);

for i = 1:1100

  % First, sample performance differences given the skills and outcomes
  
  t = nan(N,1); % contains a t_g variable for each game
  for g = 1:N   % loop over games
    s = w(G(g,1))-w(G(g,2));  % difference in skills
    t(g) = randn()+s;         % performace difference sample
    while t(g) < 0  % rejection sampling: only positive perf diffs accepted
      t(g) = randn()+s; % if rejected, sample again
    end
  end 
 
  
  % Second, jointly sample skills given the performance differences
  
  m = nan(M,1);  % container for the mean of the conditional 
                 % skill distribution given the t_g samples
  for p = 1:M
      m(p) = t'*((p==G(:,1)) - (p==G(:,2)));
  end
              
  iS = zeros(M,M); % container for the sum of precision matrices contributed
                   % by all the games (likelihood terms)
                   
  %build the iS matrix             
    for a = 1:M         
        for b = 1:a
            if a==b  
                iS(a,b) = sum(a==G(:,1)) + sum(a==G(:,2));
            else
                iS(a,b) = -sum((a==G(:,1)).*(b==G(:,2))+(a==G(:,2)).*(b==G(:,1)));
                iS(b,a) = iS(a,b);
            end
        end
    end

  iSS = diag(1./pv) + iS; % posterior precision matrix
  % prepare to sample from a multivariate Gaussian
  % Note: inv(M)*z = R\(R'\z) where R = chol(M);
  iR = chol(iSS);  % Cholesky decomposition of the posterior precision matrix
  mu = iR\(iR'\m); % equivalent to inv(iSS)*m but more efficient
    
  % sample from N(mu, inv(iSS))
  w = mu + iR\randn(M,1);
  w_plot(:,i) = w;
end
%%
w_plot_inde = w_plot(:,101:1100); %burn in
w_plot_inde = w_plot_inde(:,1:10:end); %thinning

hist = histogram(w_plot(1,:))
title('Pseudo seed 1 skill initalised as -5');
figure
ist = histogram(w_plot(6,:))
title('Pseudo seed 1 skill initalised as -5');
%%
%Plot some of the sampled player skills as a function of the Gibbs iteration
iters = linspace(1,1100,1100);
subplot(4,1,1)
plot (iters,w_plot(1,:))
xlim([0 1100])
ylim([0 4])
xlabel('Iteration')
ylabel('Skill of player 1')
subplot(4,1,2)
plot (iters,w_plot(15,:))
xlim([0 1100])
ylim([-2 2])
xlabel('Iteration')
ylabel('Skill of player 15')
subplot(4,1,3)
plot (iters,w_plot(16,:))
xlim([0 1100])
ylim([0 4])
xlabel('Iteration')
ylabel('Skill of player 16')
subplot(4,1,4)
plot (iters, w_plot(5,:))
xlim([0 1100])
ylim([0 4])
xlabel('Iteration')
ylabel('Skill of player 5')
%%
iters = linspace(1,100,100);
subplot(4,1,1)
plot (iters,w_plot_inde(1,:))
xlim([0 100])
xlabel('Iteration')
ylabel('Skill of player 1')
subplot(4,1,2)
plot (iters,w_plot_inde(15,:))
xlim([0 100])
xlabel('Iteration')
ylabel('Skill of player 15')
subplot(4,1,3)
plot (iters,w_plot_inde(16,:))
xlim([0 100])
xlabel('Iteration')
ylabel('Skill of player 16')
subplot(4,1,4)
plot (iters, w_plot_inde(5,:))
xlim([0 100])
xlabel('Iteration')
ylabel('Skill of player 5')
%%
[cov, lag] = xcov(w_plot(1,:),100,'coeff');
plot(lag,cov)
%stop being correlated when we take  approx every 6 samples, which is the 
%mixing time
m = mean(w_plot(1,:));
%%
%D1
mean_vec = mean(w_plot_inde');
var_vec = var(w_plot_inde');
inp = (mean_vec(16)-mean_vec(1))/(sqrt(var_vec(16)+var_vec(1)));
outcome_1 = normcdf(inp)

%%
%D2
player1 = w_plot_inde(1,:);
player16 = w_plot_inde(16,:);
compare = player1 <= player16;
outcome_2 = sum(compare)/length(compare)

%%
%adferd 1 se betri .. 

index = [16,1,5,11];
perc_skills = nan(4,4);

for i = 1:4
    for p = 1:4
        if index(i) == index(p)
            perc_skills(i,p) = 0;
        else
            breyta = (mean_vec(index(i))-mean_vec(index(p)))/(sqrt(var_vec(index(i))+var_vec(index(p))));
            perc_skills(i,p) = normcdf(breyta);
        end
    end
end
%%
%adferd 2 betri
index = [16,1,5,11];
perc_skills2 = nan(4,4);
for i = 1:4
    for p = 1:4
        if index(i) == index(p)
            perc_skills2(i,p) = 0;
        else
            p1 = w_plot_inde(index(i),:);
            p2 = w_plot_inde(index(p),:);
            comp = p1 > p2;
            perc_skills2(i,p) = sum(comp)/length(comp);
        end
    end
end
%%
%e
%Now compute 107x107 table, make everyone play everyone
perc_wins_gibbs = nan(107,107);

for i = 1:107
    for p = 1:107
        if i == p
            perc_wins_gibbs(i,p) = 0;
        else
            variable = (mean_vec(i)-mean_vec(p))/(sqrt(1+var_vec(i)+var_vec(p)));
            perc_wins_gibbs(i,p) = normcdf(variable);
        end
    end
end

ranks_gibbs = mean(perc_wins_gibbs,2) 

[kk_1, ii] = sort(ranks_gibbs, 'descend');

np = 107;
barh(kk(np:-1:1))
set(gca,'YTickLabel',W(ii(np:-1:1)),'YTick',1:np,'FontSize',5)
axis([0 1 0.5 np+0.5])
title ('Ranking using outcomes from Gibbs Sampling')

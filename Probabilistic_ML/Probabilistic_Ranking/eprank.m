%clear;clc
load tennis_data

M = size(W,1);            % number of players
N = size(G,1);            % number of games in 2011 season 

psi = inline('normpdf(x)./normcdf(x)');
lambda = inline('(normpdf(x)./normcdf(x)).*( (normpdf(x)./normcdf(x)) + x)');

pv = 0.5;            % prior skill variance (prior mean is always 0)

% initialize matrices of skill marginals - means and precisions
Ms = nan(M,1); 
Ps = nan(M,1);

% initialize matrices of game to skill messages - means and precisions
Mgs = zeros(N,2); 
Pgs = zeros(N,2);

% allocate matrices of skill to game messages - means and precisions
Msg = nan(N,2); 
Psg = nan(N,2);

for iter=1:1000
  % (1) compute marginal skills 
  for p=1:M
    % precision first because it is needed for the mean update
    Ps(p) = 1/pv + sum(Pgs(G==p)); 
    Ms(p) = sum(Pgs(G==p).*Mgs(G==p))./Ps(p);
  end
  
  % mean matrix/iteration
  M1(:,iter) = Ms;
  % Covar matrix/iteration
  V1(:,iter) = Ps;
  
  if iter > 1
      if norm(M1(:,iter)-M1(:,iter-1))<0.001 || norm(V1(:,iter)-V1(:,iter-1))<0.001;
          break;
      end
  end
 
  % (2) compute skill to game messages
  % precision first because it is needed for the mean update
  Psg = Ps(G) - Pgs;
  Msg = (Ps(G).*Ms(G) - Pgs.*Mgs)./Psg;
    
  % (3) compute game to performance messages
  vgt = 1 + sum(1./Psg, 2);
  mgt = Msg(:,1) - Msg(:,2); % player 1 always wins the way we store data
   
  % (4) approximate the marginal on performance differences
  Mt = mgt + sqrt(vgt).*psi(mgt./sqrt(vgt));
  Pt = 1./( vgt.*( 1-lambda(mgt./sqrt(vgt)) ) );
    
  % (5) compute performance to game messages
  ptg = Pt - 1./vgt;
  mtg = (Mt.*Pt - mgt./vgt)./ptg;   
    
  % (6) compute game to skills messages
  Pgs = 1./(1 + repmat(1./ptg,1,2) + 1./Psg(:,[2 1]));
  Mgs = [mtg, -mtg] + Msg(:,[2 1]);
end

%D
PS = 1./Ps;
varr = 1./V1;
%%
Means = zeros(4,1);
Variances = zeros(4,1);
means(1) = Ms(16,:);
variances(1) = PS(16,:);
means(2) = Ms(1,:);
variances(2) = PS(1,:);
means(3) = Ms(5,:);
variances(3) = PS(5,:);
means(4) = Ms(11,:);
variances(4) = PS(11,:);

perc_skills = nan(4,4);
perc_win = nan(4,4);

for p = 1:4
    for m = 1:4
        if p == m
            perc_skills(p,m) = 0;
            perc_win(p,m) = 0;
        else
            breyta = (means(p)-means(m))/(sqrt(variances(p)+variances(m)));
            breyta2 = (means(p)-means(m))/(sqrt(1+variances(p)+variances(m)));
            perc_skills(p,m) = normcdf(breyta);
            perc_win(p,m) = normcdf(breyta2);            
        end
    end
end

%%
%E
perc_wins_ep = nan(107,107);

for p = 1:107
    for m = 1:107
        if p == m
            perc_wins_ep(p,m) = 0;
        else
            variable = (Ms(p)-Ms(m))/(sqrt(1+PS(p)+PS(m)));
            perc_wins_ep(p,m) = normcdf(variable);            
        end
    end
end

ranks_ep = mean(perc_wins_ep,2) 

[kk_2, ii] = sort(ranks_ep, 'descend');

% np = 107;
% barh(kk(np:-1:1))
% set(gca,'YTickLabel',W(ii(np:-1:1)),'YTick',1:np,'FontSize',5)
% axis([0 1 0.5 np+0.5])
% title ('Ranking using outcomes from EP')



clear; clc
cliffworld
%%
%a)
[v_valueit, pi_valueit] = valueIteration(model,1000)
plotVP( v_valueit, pi_valueit, paramSet)
%%
%b)
% I should compare time it takes for each of them to run for all worlds
% interesting to see if other is faster for smaller world but slower for
% larger
[v_policyit, pi_policyit] = policyIteration(model,2000)
plotVP(v_policyit,pi_policyit,paramSet)
%%
%c)
[v_sarsa, pi_sarsa, cum_r_sarsa] = sarsa(model, 10000, 5000)
plotVP(v_sarsa,pi_sarsa,paramSet)
%%
%d)
[v_q, pi_q, cum_r_q] = qLearning(model, 10000, 5000)
plotVP(v_q,pi_q,paramSet)

%% Obtain plot for Part e
% Set parameters common to both algorithms
close all;
maxiters = 500;
maxeps = 500;
alpha = 0.5;
decay_rate = 1;
epsilon = 0.1;
flag = 'fixed';
% 20 runs of SARSA and Q-learning
nruns = 50;
rewardsSARSA = zeros(nruns,maxeps);
rewardsQlearning = zeros(nruns,maxeps);
for i=1:nruns
%[vSARSA,piSARSA,~,occupancySARSA,~,rewSARSA] = sarsaJ(model,maxiters,maxeps,alpha,flag,decay_rate,epsilon);
%[vQ,piQ,~,occupancyQlearning,~,rewQlearning] = qLearningJ(model,maxiters,maxeps),alpha,flag,decay_rate,epsilon);
[vSARSA, piSARSA, rewSARSA] = sarsa(model, maxiters, maxeps);
[vQ, piQ, rewQlearning] = qLearning(model, maxiters, maxeps);
rewardsSARSA(i,:)=rewSARSA';
rewardsQlearning(i,:)=rewQlearning';
end
rewardSARSA=mean(rewardsSARSA);              % Average over SARSA runs
rewardQlearning=mean(rewardsQlearning);      % Average over q-learning runs
% Smooth results with a moving average filter
smoothstride = 10;
rewardSARSA = movmean(rewardSARSA,smoothstride);
rewardQlearning = movmean(rewardQlearning,smoothstride);
% Plot cumulative rewards for the two algorithms
figure(1)
x = 1:1:maxeps; 
plot(x,rewardSARSA,x,rewardQlearning)
xlabel('Episodes','Interpreter','latex','FontSize',16)
ylabel('Cumulative reward','Interpreter','latex','FontSize',16)
legend('SARSA','Qlearning','Interpreter','latex')
yupper = 20; ylower = -100;
ylim([ylower,yupper])
xlim([1,maxeps])
grid on
% Plot value function for SARSA
% figure(2)
% plotVP(vSARSA,piSARSA,paramset)
% figure(3)
% plotVP(vQ,piQ,paramset)
function [v, pi,cum_reward_total] = qLearning(model, maxit, maxeps)


% initialize the value function
Q = zeros(model.stateCount, 4);
cum_reward_total = zeros(maxeps,1); %stores cumulative reward

alpha = 0.5;
epsilon = 0.1 ;

for i = 1:maxeps %repeat for each episode
    
    cum_reward = 0;
    
    %epsilon = 1/(i); %decrease our epsilon each episode
    %alpha = 1/((i)^0.5); %decrease alpha each episode, slower than epsilon

    s = model.startState; % every time we reset the episode, start at the given startStats
    %pick action, epsilon gready:
    
    for j = 1:maxit %repeat for each step of the episode
        
        %pick action, epsilon gready:
        if rand < (1-epsilon)
            [val,a] = max(Q(s,:));
        else
            a = randi(4);
        end
        
        %Take action and sample the state given action
        p = 0;
        random = rand;
        for s_ = 1:model.stateCount
            p = p + model.P(s, s_, a);
            if random <= p
                break
            end
        end
        % s_ should now be the next sampled state.
        r = model.R(s,a); %observe the reward 
        
        %update Q
        Q(s,a) = Q(s,a) + alpha*(r+model.gamma*max(Q(s_,:))-Q(s,a)); %update Q function
        cum_reward = cum_reward + r;
        s = s_;

        if s == model.stateCount
            break %break out of the loop of we've reached the goal state 
        end     
    end
    
    cum_reward_total(i) = cum_reward;
    
end

v = zeros(model.stateCount, 1);
pi = ones(model.stateCount, 1);
for s=1:model.stateCount
    [v(s),pi(s)] = max(Q(s,:));
end

% v = max(Q,[],2); %define the value function
% pi = ones(model.stateCount, 1);
% for s = 1:model.stateCount
%     values = zeros(4,1)
%     for a = 1:4
%         values(a) = model.P(s,:,a)*v
%         [value, indx] = max(values)
%         pi(s) = indx
%     end
% end


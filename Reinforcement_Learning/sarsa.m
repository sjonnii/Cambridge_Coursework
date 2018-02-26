function [v, pi, cum_reward_total] = sarsa(model, maxit, maxeps)

% initialize the value function
Q = zeros(model.stateCount, 4);
cum_reward_total = zeros(maxeps,1); %stores cumulative reward
% 
alpha = 0.5;
epsilon = 0.1;

for i = 1:maxeps %repeat for each episode
    
    cum_reward = 0;
    
    %epsilon = 1/(i); %decrease our epsilon each episode
    %alpha = 1/((i)^0.1); %decrease alpha each episode, slower than epsilon

        
    s = model.startState; % every time we reset the episode, start at the given startStats
    
    %pick action, epsilon gready:
    random = rand;
    if random < (1-epsilon)
        [val,a] = max(Q(s,:));
    else
        a = randi(4);
    end
    
    for j = 1:maxit %repeat for each step of the episode
        p = 0;
        random = rand;
        %Take action and sample the state given action
        for s_ = 1:model.stateCount
            p = p + model.P(s, s_, a);
            if random <= p
                break
            end
        end
        % s_ should now be the next sampled state.
        
        r = model.R(s,a); %observe the reward
        
        %Choose next action from s_, epsilon gready(our policy)
        random = rand;
        if random < (1-epsilon)
            [val,a_] = max(Q(s_,:));
        else
            a_ = randi(4);
        end
        
        %update Q
        Q(s,a) = Q(s,a) + alpha*(r+model.gamma*Q(s_,a_)-Q(s,a)); %update Q function
        s = s_;
        a = a_;
        
        cum_reward = cum_reward + r;

        if s == model.stateCount
            break %break out of the loop of we've reached the goal state    
        end
        
    end
    
    cum_reward_total(i) = cum_reward;
end

% Update value function and policy function
v = zeros(model.stateCount, 1);
pi = ones(model.stateCount, 1);
for s=1:model.stateCount
    [v(s),pi(s)] = max(Q(s,:)); 
end
% occupancy = occupancy / sum(occupancy) ;
end


function [v, pi] = valueIteration(model, maxit)

%asynchronous dynamic programming -- skoda ad nota thad til ad flyta fyrir
%in place dynamic programming
%prioritised sweeping
%real time dynamic programming
%end of lecture 3 david silver

% initialize the value function
v = zeros(model.stateCount, 1);

%initialize the policy 
pi = ones(model.stateCount, 1);
for i = 1:maxit
    % initialize the new value function
    v_ = zeros(model.stateCount, 1);

    % perform the Bellman update for each state
    %Can I eliminate the second s forloop by vectorizing? 
    for s = 1:model.stateCount
        summa = zeros(4,1);
        for s_ = 1:model.stateCount
            tmp = squeeze(model.P(s,s_,:)).*(model.R(s,:)'+model.gamma*v(s_));
            summa = summa + tmp;
        end
        v_(s) = max(summa);
    end
    if norm(abs(v-v_)) < 1e-10
       break;
    end
    v = v_;
end
% disp (v)

for s = 1:model.stateCount
    values = zeros(4,1)
    for a = 1:4
        values(a) = model.P(s,:,a)*v
        [value, indx] = max(values)
        pi(s) = indx
    end
end

    

% cols = paramSet.colCount
% for s = 1:model.stateCount
%     indx = find(sum(squeeze(model.P(s,:,:)),2));%find all possible transition states
%     [max_values,max_idx] = max(v(indx))%find the highest value
%     opt_state = indx(max_idx)%find the state that has the highest value
%     %pick action to get to this state
%     if opt_state == s-cols
%         pi(s) = 1 %best to move up
%     elseif opt_state == s+cols
%         pi(s) = 2 %best to move down
%     elseif opt_state == s-1
%         pi(s) = 3 %best to go left
%     elseif opt_state == s+1
%         pi(s) = 4 %best to go right
%     else
%         continue
%     end
% end
   
    
    
    






function [v, pi] = policyIteration(model, maxit)

% initialize the value function
v = zeros(model.stateCount, 1);
pi = ones(model.stateCount, 1);

telj2 = 0
for i = 1:maxit
    % evaluate the value function for a given policy
    boo = true;
    telj2 = telj2+1
    teljari = 0;
    while boo 
        teljari = teljari + 1;
        v_ = zeros(model.stateCount, 1);
        for s = 1:model.stateCount
            summa = zeros(4,1);
            for s_ = 1:model.stateCount
                tmp = squeeze(model.P(s,s_,:)).*(model.R(s,:)'+model.gamma*v(s_));
                summa = summa + tmp;
            end
            v_(s) = summa(pi(s));
        end
        if norm(abs(v-v_)) < 0.1 
            boo = false;
        end
        v = v_;
    end
    %greedily pick policy based on value function
    for s = 1:model.stateCount
        values = zeros(4,1);
        for a = 1:4
            values(a) = model.P(s,:,a)*v;
            [value, indx] = max(values);
            pi(s) = indx;
        end
    end


end

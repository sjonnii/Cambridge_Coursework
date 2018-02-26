clear,clc;
load kos_doc_data.mat

W = max([A(:,2); B(:,2)]);  % number of unique words
D = max(A(:,1));            % number of documents in A
K = 20;                     % number of mixture components we will use

alpha = 10;                 % parameter of the Dirichlet over mixture components
gamma = 0.1;                % parameter of the Dirichlet over words
%%
for w = 1:W
    c(w) = sum(A(A(:,2)==w,3));
end

[counts, indicies] = maxk(c,20)
top20 = V(indicies)

barh (counts./length(A))
set(gca,'YTickLabel',V([indicies]),'Ytick',1:20)

%%
counts_alphas = c' + 0.1*ones(W, 1);
probabilities = counts_alphas ./ (sum(counts_alphas));

%%
doc = B(B(:,1) == 2001,2);%ID of all words in the document
doc_W = B(B(:,1) == 2001,3);%Counts of corresponding ID's
log_prob = 0;
for i = 1:length(doc)
    log_prob = log_prob + log(probabilities(doc(i)))*doc_W(i);
end
perplexity = exp((-1/sum(doc_W)) * log_prob)
% 
id_total = B(:,2);
counts_total = B(:,3);
log_prob_total = 0;
for i = 1:length(id_total)
    log_prob_total = log_prob_total + counts_total(i)*log(probabilities(id_total(i)));
end

perplexity_total = exp((-1/sum(counts_total)) * log_prob_total);

%%


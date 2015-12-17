function [sparsep,ts] = SparseLabeledPos(densep)

%[nlandmarks,nd,T] = size(densep);
ts = find(any(any(~isnan(densep),1),2));
sparsep = densep(:,:,ts);

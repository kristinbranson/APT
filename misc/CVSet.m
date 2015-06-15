function cvidx = CVSet(expidx,ncrossvalsets)

% choose cross-validation set memberships
[uniqueidx,~,expidx1] = unique(expidx);
nuniqueidx = numel(uniqueidx);
order = randperm(nuniqueidx);
expidx1 = order(expidx1);
counts = hist(expidx1,1:nuniqueidx);
N = numel(expidx);
s = [0,cumsum(counts)];
j = 1;
cvidx = nan(1,N);

for i = 1:ncrossvalsets,
  
  nperset = round((N-s(j))/(ncrossvalsets-i+1));
  [~,k] = min(abs(s(j+1:end)-s(j) - nperset));
  k = j + k;
  cvidx(ismember(expidx1,j:k-1)) = i;
  j = k;
  
end


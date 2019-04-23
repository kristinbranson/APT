% sample = samplep(p,n)
% returns n samples, where the probability of i is p(i) for 1 <=i
% <= length(p), 0 otherwise. 

function s = samplep(p,n)

% normalize, just in case
p = p(:);
p = normalizep(p);
p(end) = 1;
m = length(p);

% compute the cumulative sum of probabilities
c = cumsum(p);

% choose n random numbers uniformly between 0 and 1
r = rand(1,n);

% find the first element of c >= r
s = findfirstdim(repmat(c,[1,n])>=repmat(r,[m,1]),1);
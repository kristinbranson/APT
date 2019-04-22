function [v,xbest,sbest] = cross_entropy_method(score,generate,maximize,v,rho,varargin)

maxniters = 100;
d = 5;
N = 50;
epsilon = .00001;
iter0 = 1;
sbest = -inf;
for i = 1:2:length(varargin),
  switch varargin{i},
   case 'maxniters',
    maxniters = varargin{i+1};
   case 'niters no change',
    d = varargin{i+1};
   case 'nsamples',
    N = varargin{i+1};
   case 'epsilon',
    epsilon = varargin{i+1};
   case 'restart',
     iter0 = varargin{i+1}.i+1;
     sbest = varargin{i+1}.sbest;
     v = varargin{i+1}.v;
     xbest = varargin{i+1}.xbest;
  end;
end;

gamma = nan*ones(d,1);
n = ceil((1-rho)*N);

for i = iter0:maxniters,
  
  fprintf('%d ',i);
  % generate N sample parameters
  x = generate(v,N);
  % score the samples
  s = score(x);
  % sort based on score
  [ssorted,sorder] = sort(-s);
  x = x(:,sorder);
  s = s(sorder);
  % choose the ml parameters for the top scores
  ncurr = find(s < s(n),1);
  if isempty(ncurr), 
    ncurr = length(s);
  else
    ncurr = ncurr - 1; 
  end
  v = maximize(x(:,1:ncurr));
   
  % save the current worst elite score
  gamma(1:d-1) = gamma(2:d);
  gamma(d) = s(n);
  fprintf(' %f\n',gamma(d));
  
  % no decrease in gamma in past d iterations?
  if ((gamma(d) <= gamma(1)) || ...
      all((gamma <= mean(gamma) + epsilon) & ...
	  (gamma >= mean(gamma) - epsilon))),
    break;
  end;
  
  if s(1) > sbest,
    sbest = s(1);
    xbest = x(:,1);
  end  
  
  save cross_entropy_method_save.mat sbest xbest v i;
  
end

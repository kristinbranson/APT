%% function mypca
% [mu,pc_vec,pc_val,pc_rec,pc_coeff,err] = MYPCA(data,...)
% 
% This function performs principal component analysis on the input data. It uses
% the Turk and Pentland trick to speed up computation if the dimensionality is
% larger than the number of samples. 
% 
% Input:
% data: data is a matrix of any number of dimensions greater than 1. The first
% d-1 dimensions are treated as features of the data, and the last dimension is
% the number of data points. Thus, to perform PCA, we resize the data of size sz
% to be of size [prod(sz(1:end-1)),sz(end)]. 
% 
% Optional Inputs:
% optional inputs can be used to determine the number of principal components
% returned. Any number of constraints can be specified. The minimum number of
% principal components satisfying ANY constraint is returned. 
% 'maxnpc': The maximum number of PCs to return. 
% 'maxrecerr': The maximum maximum reconstruction error of any data point. 
% 'averecerr': The maximum average reconstruction error of the data points.
% 'medrecerr': The maximum median reconstruction error of the data points. 
% 'varfrac': The minimum fraction of the variance that must be captured by the
% reconstruction. 
%
% Outputs:
% mu: The mean of the data. mu is of size [sz(1:end-1),1].
% pc_vec: The principal components of the data. pc_vec is of size
% [sz(1:end-1),npc], where npc is the number of principal components chosen
% automatically. 
% pc_val: The eigenvalues for each returned principal component. pc_val is npc x
% 1. 
% pc_rec: The reconstruction of the data using the principal components. pc_rec
% is of size sz. 
% pc_coeff: The projecton of the data onto the principal subspace. pc_coeff(i,j)
% is the projection of data point j on principal component i. 
%
function [mu,pc_vec,pc_val,pc_rec,pc_coeff,err] = mypca(data,varargin)

[maxnpc,maxrecerr,averecerr,medrecerr,minvarfrac,zscore] = ...
  myparse(varargin,'maxnpc',[],'maxrecerr',[],'averecerr',[],'medrecerr',[],...
  'varfrac',[],'zscore',0);

MINEIGVAL = .00001;

% input size of data
datasz = size(data);
% dimensions
d = prod(datasz(1:end-1));
% number of training points
n = datasz(end);

% reshape data so that it is prod(datasz(1:end-1)) x datasz(end)
data = reshape(data,d,n);

% compute the mean
mu = mean(data,2);

if zscore,
  data = data ./ repmat(std(data,1,2),[1,n]);
end;

% subtract off the mean
data = data - repmat(mu,[1,n]);

% do we want to use the turk and pentland trick?
if d > n,
  
  % yes, use the turk and pentland trick

  % compute the n x n matrix
  L = data'*data;

  % compute the eigenvectors of this small matrix
  [pc_vec,pc_val] = eig(L);
  
  % convert the eigenvectors of data'*data into eigenvectors of
  % data*data'
  pc_vec = data*pc_vec;
  
else

  % no, just compute eigenvectors of covariance directly
  
  % covariance
  L = data*data';

  % compute the eigenvectors of the covariance
  [pc_vec,pc_val] = eig(L);

end;

% remove eigenvectors with eigenvalue very small
pc_val = diag(pc_val);
smallinds = find(pc_val <= MINEIGVAL);
pc_vec(:,smallinds) = [];
pc_val(smallinds) = [];

% sort the eigenvectors from largest to smallest
[pc_val,order] = sort(pc_val,'descend');
pc_vec = pc_vec(:,order);

npc = length(pc_val);

% normalize
for i = 1:npc,
  pc_vec(:,i) = pc_vec(:,i) / norm(pc_vec(:,i));
end;

%% how many principal components do we keep?

% if there is a maximum number input, restrict
if ~isempty(maxnpc),
  npc = min(maxnpc,npc);
end;

%% if there is a maximum reconstruction error input, restrict

if ~isempty(maxrecerr) || ~isempty(averecerr) || ...
    ~isempty(medrecerr) || ~isempty(minvarfrac),

  % compute the variance of the data
  datavar = sum(var(data,1,2));
  
  % compute the reconstruction using 1:npc pcs until error is small enough
  pc_rec = repmat(mu,[1,n]);
  pc_coeff = zeros(npc,n);

  for i = 1:npc,
    % coefficient of ith pc
    pc_coeff(i,:) = pc_vec(:,i)'*data;
    % reconstruction using i pcs
    pc_rec = pc_rec + repmat(pc_coeff(i,:),[d,1]).*repmat(pc_vec(:,i),[1,n]);
    % error using i pcs
    err = sum((pc_rec - data).^2,1);
    % variance using i pcs
    recvar = sum(var(pc_rec,1,2));
    
    % check maximum reconstruction error
    if ~isempty(maxrecerr),
      if max(err) <= maxrecerr,
        break;
      end;
    end;
  
    % check mean reconstruction error
    if ~isempty(averecerr),
      if mean(err) <= averecerr,
        break;
      end;
    end;

    % check median reconstruction error
    if ~isempty(medrecerr),
      if median(err) <= medrecerr,
        break;
      end;
    end;    
    
    if ~isempty(minvarfrac),
      if recvar / datavar >= minvarfrac,
        break;
      end;
    end;
  end;
  
  npc = i;
  pc_coeff = pc_coeff(1:npc,:);
  
end;

% keep the top pcs
pc_vec = pc_vec(:,1:npc);
pc_val = pc_val(1:npc);

% compute reconstruction if necessary and we haven't done so already
if ~exist('pc_rec','var') && nargout >= 4,
  pc_rec = repmat(mu,[1,n]);
  pc_coeff = zeros(npc,n);

  for i = 1:npc,
    % coefficient of ith pc
    pc_coeff(i,:) = pc_vec(:,i)'*data;
    % reconstruction using i pcs
    pc_rec = pc_rec + repmat(pc_coeff(i,:),[d,1]).*repmat(pc_vec(:,i),[1,n]);
  end;
  err = sum((pc_rec - data).^2,1);

end;

% reshape the principal components, etc
pc_vec = reshape(pc_vec,[datasz(1:end-1),npc]);
mu = reshape(mu,[datasz(1:end-1),1]);
if exist('pc_rec','var'),
  pc_rec = reshape(pc_rec,[datasz(1:end-1),n]);
end;

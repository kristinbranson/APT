function mix = mygmminit(x,centres,idx,covartype)
%GMMINIT Initialises Gaussian mixture model from data
%
[ndata, xdim] = size(x);
k = rows(centres);
mix = gmm(xdim,k,covartype);
I = eye(k);
post = I(idx,:);
mix.centres = centres;

% Arbitrary width used if variance collapses to zero: make it 'large' so
% that centre is responsible for a reasonable number of points.
GMM_WIDTH = 1.0;

% Set priors depending on number of points in each cluster
cluster_sizes = max(sum(post, 1), 1);  % Make sure that no prior is zero
mix.priors = cluster_sizes/sum(cluster_sizes); % Normalise priors

switch mix.covar_type
case 'spherical'
   if mix.ncentres > 1
      % Determine widths as distance to nearest centre 
      % (or a constant if this is zero)
      for i = 1:mix.ncentres
        mix.covars(i) = mean(dist2(x(idx==i,:),mix.centres(i,:)));
      end
   else
      % Just use variance of all data points averaged over all
      % dimensions
      mix.covars = mean(diag(cov(x)));
   end
  case 'diag'
    for j = 1:mix.ncentres
      % Pick out data points belonging to this centre
      c = x(find(post(:, j)),:);
      diffs = c - (ones(size(c, 1), 1) * mix.centres(j, :));
      mix.covars(j, :) = sum((diffs.*diffs), 1)/size(c, 1);
      % Replace small entries by GMM_WIDTH value
      mix.covars(j, :) = mix.covars(j, :) + GMM_WIDTH.*(mix.covars(j, :)<eps);
    end
  case 'full'
    for j = 1:mix.ncentres
      % Pick out data points belonging to this centre
      c = x(find(post(:, j)),:);
      diffs = c - (ones(size(c, 1), 1) * mix.centres(j, :));
      mix.covars(:,:,j) = (diffs'*diffs)/(size(c, 1));
      % Add GMM_WIDTH*Identity to rank-deficient covariance matrices
      if rank(mix.covars(:,:,j)) < mix.nin
	mix.covars(:,:,j) = mix.covars(:,:,j) + GMM_WIDTH.*eye(mix.nin);
      end
    end
  case 'ppca'
    for j = 1:mix.ncentres
      % Pick out data points belonging to this centre
      c = x(find(post(:,j)),:);
      diffs = c - (ones(size(c, 1), 1) * mix.centres(j, :));
      [tempcovars, tempU, templambda] = ...
	ppca((diffs'*diffs)/size(c, 1), mix.ppca_dim);
      if length(templambda) ~= mix.ppca_dim
	error('Unable to extract enough components');
      else 
        mix.covars(j) = tempcovars;
        mix.U(:, :, j) = tempU;
        mix.lambda(j, :) = templambda;
      end
    end
  otherwise
    error(['Unknown covariance type ', mix.covar_type]);
end


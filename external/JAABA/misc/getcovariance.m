function S = getcovariance(mix,inds,centers)

if ~exist('inds','var'),
  inds = 1:mix.nin;
end;
if ~exist('centers','var'),
  centers = 1:mix.ncentres;
end;

inds = inds(:)';
ninds = length(inds);
centers = centers(:)';
ncenters = length(centers);
S = zeros(ninds,ninds,ncenters);

switch mix.covar_type,

 case 'spherical',
  for i = 1:ncenters,
    j = centers(i);
    S(:,:,i) = eye(ninds)*mix.covars(j);
  end;

 case 'diag',
  for i = 1:ncenters,
    j = centers(i);
    S(:,:,i) = diag(mix.covars(j,:));
  end;
  
 case {'full','sparse'},
  S = mix.covars(inds,inds,centers);

 case 'ppca',
  for i = 1:ncenters,
    j = centers(i);
    covars = mix.covars(j) * eye(mix.nin) + ...
	     mix.U(:, :, j)*(diag(mix.lambda(j, :))-diag(mix.covars(j)))* ...
	     (mix.U(:, :, j)');
    S(:,:,i) = covars(inds,inds);
  end;

 case 'zero',
  S(:) = 0;

 otherwise,
  error(['Unknown covar_type ',mix.covar_type]);
end;
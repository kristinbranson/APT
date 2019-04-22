function [inds,inds_bin] = fernsInds2( data, fids, thrs )
% Compute indices for each input by each fern.
%
% USAGE
%  inds = fernsInds( data, fids, thrs )
%
% INPUTS
%  data     - [NxF] N length F binary feature vectors
%  fids     - [MxS] feature ids for each fern for each depth
%  thrs     - [MxS] threshold corresponding to each fid
%
% OUTPUTS
%  inds     - [NxM] computed indices for each input by each fern
%
% EXAMPLE
%
% See also fernsClfTrain, fernsClfApply
%
% Piotr's Image&Video Toolbox      Version 2.50
% Copyright 2010 Piotr Dollar.  [pdollar-at-caltech.edu]
% Please email me if you find bugs, or have suggestions or questions!
% Licensed under the Lesser GPL [see external/lgpl.txt]

% do this without a mex file for simplicity
[M,S]=size(fids); N=size(data,1);
inds_bin = bsxfun(@lt,data(:,fids),reshape(thrs,[1,M*S]));
inds_bin = reshape(inds_bin,[N,M,S]);
pows = reshape(uint32(2.^(S-1:-1:0)),[1,1,S]);
inds = 1+sum(bsxfun(@times,uint32(inds_bin),pows),3);

%%% OLD MATLAB CODE -- NOW IN MEX
% [M,S]=size(fids); N=size(data,1);
% inds = zeros(N,M,'uint32');
% for n=1:N
%   for m=1:M
%     for s=1:S
%       inds(n,m)=inds(n,m)*2;
%       if( data(n,fids(m,s))<thrs(m,s) )
%         inds(n,m)=inds(n,m)+1;
%       end
%     end
%   end
% end
% inds=inds+1;
% end

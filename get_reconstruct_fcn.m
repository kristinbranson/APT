function reconstructfun = get_reconstruct_fcn(caldata,usegeometricerror)

% [XYZ,uv_re,err] = reconstructfun(uv)
% reconstructfun(uv) takes input uv which is [2 x nviews x n]
% and produces outputs:
% XYZ: [3 x n] 3-d reconstructed points
% uv_re: [2 x nviews x n] reprojections
% err: [1 x n] reprojection error per view

if nargin < 2,
  usegeometricerror = true;
end

if isa(caldata,'CalRigSH'),
  
  if isfield(caldata.kineData,'DLT_1'),
    dlt_side = caldata.kineData.DLT_1;
    dlt_front = caldata.kineData.DLT_2;
  elseif isfield(caldata.kineData,'cal')
    dlt_side = caldata.kineData.cal.coeff.DLT_1;
    dlt_front = caldata.kineData.cal.coeff.DLT_2;
  else
    error('Not implemented');
  end
  reconstructfun = @(varargin) CalRigSH_reconstruct_fcn(dlt_side,dlt_front,usegeometricerror,varargin{:});
  
elseif isa(caldata,'OrthoCamCalPair'),
  reconstructfun = @(varargin) OrthoCamCalPair_reconstruct_fcn(caldata,varargin{:});
else
  error('Not implemented');
end

function [XYZ,uv_re,err] = OrthoCamCalPair_reconstruct_fcn(caldata,uv,varargin)

sz = size(uv);
nviews = sz(2);
d = sz(1);
assert(nviews==2);
assert(d==2);
n = prod([1,sz(3:end)]);
uv_re = uv;

[XYZ,err,uv_re(:,1,:),uv_re(:,2,:)] = caldata.stereoTriangulate(reshape(uv(:,1,:),[d,n]),reshape(uv(:,1,:),[d,n]),varargin{:});

function [XYZ,uv_re,err] = CalRigSH_reconstruct_fcn(dlt_side,dlt_front,usegeometricerror,uv,varargin)

[S,leftovers] = myparse_nocheck(varargin,'S',[]);

sz = size(uv);
nviews = sz(2);
d = sz(1);
assert(nviews==2);
assert(d==2);
n = prod([1,sz(3:end)]);
uv_re = uv;
xside = reshape(uv(:,1,:),[d,n]);
xfront = reshape(uv(:,2,:),[d,n]);

if ~isempty(S),
  Sside = reshape(S(:,:,1,:),[d,d,n]);
  Sfront = reshape(S(:,:,2,:),[d,d,n]);
  leftovers = [leftovers,{'Sside',Sside,'Sfront',Sfront}];
end

if true,%~usegeometricerror,
  [XYZ,err,~,~,uv_re(:,2,:),uv_re(:,1,:)] = dlt_2D_to_3D_point_vectorized(dlt_front,dlt_side,xfront,xside,'geometricerror',usegeometricerror,leftovers{:});
else

  XYZ = nan([3,sz(3:end)]);
  err = nan([1,sz(3:end)]);
  
  for i = 1:n,
    [XYZ(:,i),err(i),~,~,xfront_re,xside_re] = dlt_2D_to_3D_point(dlt_front,dlt_side,xfront(:,i),xside(:,i),'geometricerror',usegeometricerror,leftovers{:});
    uv_re(:,1,i) = xside_re;
    uv_re(:,2,i) = xfront_re;
  end
  
  % uv_re(:,1,:) = xside_re;
  % uv_re(:,2,:) = xfront_re;
end
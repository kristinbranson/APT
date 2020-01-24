function projectfun = get_project_fcn(caldata)

% [XYZ,uv_re,err] = reconstructfun(uv)
% reconstructfun(uv) takes input uv which is [2 x nviews x n]
% and produces outputs:
% XYZ: [3 x n] 3-d reconstructed points
% uv_re: [2 x nviews x n] reprojections
% err: [1 x n] reprojection error per view

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
  projectfun = @(varargin) CalRigSH_project_fcn(dlt_side,dlt_front,varargin{:});
  
elseif isa(caldata,'OrthoCamCalPair'),
  projectfun = @(varargin) OrthoCamCalPair_project_fcn(caldata,varargin{:});
else
  error('Not implemented');
end

function [XYZ,uv_re,err] = OrthoCamCalPair_project_fcn(caldata,uv,varargin)

error('TODO');

sz = size(uv);
nviews = sz(2);
d = sz(1);
assert(nviews==2);
assert(d==2);
n = prod([1,sz(3:end)]);
uv_re = uv;

[XYZ,err,uv_re(:,1,:),uv_re(:,2,:)] = caldata.stereoTriangulate(reshape(uv(:,1,:),[d,n]),reshape(uv(:,1,:),[d,n]),varargin{:});

function [uv_re] = CalRigSH_project_fcn(dlt_side,dlt_front,XYZ)

sz = size(XYZ);
nviews = 2;
d = sz(1);
assert(d==3);
n = prod([1,sz(2:end)]);
uv_re = nan([2,nviews,n]);

[uv_re(1,2,:),uv_re(2,2,:)] = dlt_3D_to_2D(dlt_front,XYZ(1,:),XYZ(2,:),XYZ(3,:));
[uv_re(1,1,:),uv_re(2,1,:)] = dlt_3D_to_2D(dlt_side,XYZ(1,:),XYZ(2,:),XYZ(3,:));


function varargout = shapeGt( action, varargin )
%
% Wrapper with utils for handling shape as list of landmarks
%
% shapeGt contains a number of utility functions, accessed using:
%  outputs = shapeGt( 'action', inputs );
%
% USAGE
%  varargout = shapeGt( action, varargin );
%
% INPUTS
%  action     - string specifying action
%  varargin   - depends on action
%
% OUTPUTS
%  varargout  - depends on action
%
% FUNCTION LIST
% 
%%%% Model creation and visualization %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%   shapeGt>createModel, shapeGt>draw
%
%%%% Shape composition, inverse, distances, projection %%%%%%%%%%%%%%%
% 
%   shapeGt>compose,shapeGt>inverse, shapeGt>dif, shapeGt>dist
%   shapeGt>compPhiStar, shapeGt>reprojectPose, shapeGt>projectPose
% 
%%%% Shape-indexed features computation %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   
%   shapeGt>ftrsGenIm,shapeGt>ftrsCompIm
%   shapeGt>ftrsGenDup,shapeGt>ftrsComDup
%   shapeGt>ftrsOcclMasks, shapeGt>codifyPos
%   shapeGt>getLinePoint, shapeGt>getSzIm
%   
%%%%% Random shape generation for initialization  %%%%%%%%%%%%%%%%%%%%
%   
%   shapeGt>initTr, shapeGt>initTest
%
% EXAMPLES
%
%%create COFW model 
%   model = shapeGt( 'createModel', 'cofw' );
%%draw shape on top of image
%   shapeGt( 'draw',model,Image,shape);
%%compute distance between two set of shapes (phis1 and phis2)
%   d = shapeGt( 'dist',model,phis1,phis2);
%
% For full function example usage, see individual function help and how 
%  they are used in:  demoRCPR, FULL_demoRCPR, rcprTrain, rcprTest
%
% Copyright 2013 X.P. Burgos-Artizzu, P.Perona and Piotr Dollar.  
%  [xpburgos-at-gmail-dot-com]
% Please email me if you find bugs, or have suggestions or questions!
% Licensed under the Simplified BSD License [see bsd.txt]
%
%  Please cite our paper if you use the code:
%  Robust face landmark estimation under occlusion, 
%  X.P. Burgos-Artizzu, P. Perona, P. Dollar (c)
%  ICCV'13, Sydney, Australia

% Modified by Allen Lee, Kristin Branson

varargout = cell(1,max(1,nargout));
[varargout{:}] = feval(action,varargin{:});
end

function model = createModel( type, d, nfids, nviews ) %#ok<*DEFNU>
% Create shape model (model is necessary for all other actions).
model=struct('nfids',0,'D',0,'isFace',1,'name',type);
switch type
    case 'cofw' % COFW dataset (29 landmarks: X,Y,V)
        model.d=3;model.nfids=29;model.nviews=1;
    case 'lfpw' % LFPW dataset (29 landmarks: X,Y)
        model.d=2;model.nfids=29;model.nviews=1;
    case 'helen' % HELEN dataset (194 landmarks: X,Y)
        model.d=2;model.nfids=194;model.nviews=1;
    case 'lfw' % LFW dataset (10 landmarks: X,Y)
        model.d=2;model.nfids=10;model.nviews=1;
    case 'pie' %Multi-pie & 300-Faces in the wild dataset (68 landmarks)
        model.d=2;model.nfids=68;model.nviews=1;
    case 'apf' %anonimous portrait faces
        model.d=2;model.nfids=55;model.nviews=1;
    case 'larva'
        model.d=2;model.nfids=4;model.nviews=1;
    case 'mouse_paw'
        model.d=2;model.nfids=1;model.nviews=1;
    case 'mouse_paw2'
        model.d=2;model.nfids=2;model.nviews=1;
    case 'mouse_paw3D'
        model.d=3;model.nfids=1;model.nviews=2;
    case 'fly_RF2'
        model.d=3;model.nfids=6;model.nviews=3;
    case 'mouse_paw_multi'
        model.d=3;model.nfids=6;model.nviews=2;
    otherwise
        model.d = d; model.nfids = nfids; model.nviews=nviews;
end

if nargin >= 2 && ~isempty(d),
  if model.d ~= d,
    warning('d = %d does not match default for model type %s = %d',d,model_type,model.d);
  end
  model.d = d;
end
if nargin >= 3 && ~isempty(nfids),
  if model.nfids ~= nfids,
    warning('nfids = %d does not match default for model type %s = %d',nfids,model_type,model.nfids);
  end
  model.nfids = nfids;
end
if nargin >= 4 && ~isempty(nviews),
  if model.nviews ~= nviews,
    warning('nviews = %d does not match default for model type %s = %d',nviews,model_type,model.nviews);
  end
  model.nviews = nviews;
end

model.D = model.nfids*model.d;

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function h = draw( model, Is, phis, varargin )
% Draw shape with parameters phis using model on top of image Is.
dfs={'n',25, 'clrs','gcbm', 'drawIs',1, 'lw',10, 'is',[]};
[n,cs,drawIs,lw,is]=getPrmDflt(varargin,dfs,1);

% display I
if(drawIs), im(Is); colorbar off; axis off; title(''); axis('ij'); end%clf
% special display for face model (draw face points)
hold on,
if( isfield(model,'isFace') && model.isFace ),
    [N,D]=size(phis);
    if(strcmp(model.name,'cofw') || strcmp(model.name,'fly_RF2')),
        %WITH OCCLUSION
        nfids = D/3;
        for n=1:N
            occl=phis(n,(nfids*2)+1:nfids*3);
            vis=find(occl==0);novis=find(occl==1);
            plot(phis(n,vis),phis(n,vis+nfids),'g.',...
                'MarkerSize',lw);
            h=plot(phis(n,novis),phis(n,novis+nfids),'r.',...
                'MarkerSize',lw);
        end
    else
        %REGULAR
        if(N==1),cs='g';end, nfids = D/2;
        for n=1:N
            h=plot(phis(n,1:nfids),phis(n,nfids+1:nfids*2),[cs(n) '.'],...
                'MarkerSize',lw);
        end
    end
end
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function pos=ftrsOcclMasks(xs)
%Generate 9 occlusion masks for varied feature locations
pos=cell(9,1);
for m=1:9
    switch(m)
        case 1,pos{m}=(1:numel(xs(:,1)))';
        case 2,%top half
            pos{m}=find(xs(:,2)<=0);
        case 3,%bottom half
            pos{m}=find(xs(:,2)>0);
        case 4,%right
            pos{m}=find(xs(:,1)>=0);
        case 5,%left
            pos{m}=find(xs(:,1)<0);
        case 6,%right top diagonal
            pos{m}=find(xs(:,1)>=xs(:,2));
        case 7,%left bottom diagonal
            pos{m}=find(xs(:,1)<xs(:,2));
        case 8,%left top diagonal
            pos{m}=find(xs(:,1)*-1>=xs(:,2));
        case 9,%right bottom diagonal
            pos{m}=find(xs(:,1)*-1<xs(:,2));
    end
end
end

function ftrData = ftrsGenDup( model, varargin )
% Generate random shape indexed features, relative to
% two landmarks (points in a line, RCPR contribution)
% Features are then computed using frtsCompDup
%
% USAGE
%  ftrData = ftrsGenDup( model, varargin )
%
% INPUTS
%  model    - shape model (see createModel())
%  varargin - additional params (struct or name/value pairs)
%   .type     - [2] feature type (1 or 2)
%   .F        - [100] number of ftrs to generate
%   .radius   - [2] sample initial x from circle of given radius
%   .nChn     - [1] number of image channels (e.g. 3 for color images)
%   .pids     - [] part ids for each x
%
% OUTPUTS
%  ftrData  - struct containing generated ftrs
%   .type     - feature type (1 or 2)
%   .F        - total number of features
%   .nChn     - number of image channels
%   .xs       - feature locations relative to unit circle
%   .pids     - part ids for each x
%
% EXAMPLE
%
% See also shapeGt>ftrsCompDup

dfs={'type',4,'F',20,'radius',1,'nChn',3,'pids',[],'mask',[]};
[type,F,radius,nChn,pids,mask]=getPrmDflt(varargin,dfs,1);
F2=max(100,ceil(F*1.5));
xs=[];nfids=model.nfids;
while(size(xs,1)<F),
    %select two random landmarks
    xs(:,1:2)=randint2(F2,2,[1 nfids]);
    %make sure they are not the same
    neq = (xs(:,1)~=xs(:,2));
    xs=xs(neq,:);
end
xs=xs(1:F,:);
%select position in line
xs(:,3)=(2*radius*rand(F,1))-radius;
if(nChn>1),
    if(type==4),%make sure subbtractions occur inside same channel
        chns = randint2(F/2,1,[1 nChn]);
        xs(1:2:end,4) = chns; xs(2:2:end,4) = chns;
    else xs(:,4)=randint2(F,1,[1 nChn]);
    end
end
if(isempty(pids)), pids=floor(linspace(0,F,2)); end
ftrData=struct('type',type,'F',F,'nChn',nChn,'xs',xs,'pids',pids);
end

%#3DOK
function ftrData = ftrsGenDup2( model, varargin )
% Generate random shape indexed features, relative to
% two landmarks outside the line that connects them (JRG contribution)
% Features are then computed using frtsCompDup2
%
% INPUTS
%  model    - shape model (see createModel())
%  varargin - additional params (struct or name/value pairs), see below
%
% OUTPUTS
%  ftrData  - struct containing generated ftrs
%   .type     - feature type (1 or 2)
%   .F        - total number of features
%   .nChn     - number of image channels
%   .xs       - feature locations relative to unit circle, xs(:,4) is the
%               angle between the line that connects the two landmarks and
%               the feature point
%   .pids     - part ids for each x
%
% EXAMPLE
%
% See also shapeGt>ftrsCompDup

dfs = {...
  'type','2lm',... % feature type
  'F',20,... % number of features to generate
  'radius',1,... % radius or radius factor for sampling within neighborhood of certain size
  'abratio',2,... 
  'nChn',3,... % number of image channels
  'pids',[],... % "part ids for each x"
  'randctr',false,... % see Features.generate*
  'neighbors',{},... % see Features.generate*
  'fids',[]}; % see Features.generate*
[type,F,radius,abratio,nChn,pids,randctr,neighbors,fids] = ...
  getPrmDflt(varargin,dfs,0);

switch type
  case 'single landmark'
    % features generated relative to a single landmark
    xs = Features.generate1LM(model,'F',F,'radius',radius);
  case '2lm'
    % features generated relative to (eg the line segment connecting) two 
    % landmarks 
    xs = Features.generate2LM(model,'F',F,'radiusFac',radius,...
      'randctr',randctr,'neighbors',neighbors,'fids',fids,'nchan',nChn);
  case 'two landmark elliptical'
    xs = Features.generate2LMelliptical(model,'F',F,'radiusFac',radius,...
      'abratio',abratio,'nchan',nChn);
  case '2lmdiff'
    % differences between 2 features in the same channel, where features
    % are generated relative to two landmarks
    xs = Features.generate2LMDiff(model,'F',F,'radiusFac',radius,...
      'randctr',randctr,'neighbors',neighbors,'fids',fids,'nchan',nChn);
  otherwise
    assert(false,'unknown type');
end
    
if isempty(pids)
  pids = floor(linspace(0,F,2)); 
end
ftrData = struct('type',type,'F',F,'nChn',nChn,'xs',xs,'pids',pids);
end

function ftrData = ftrsGenKBOrig( model, varargin )
% hack for Jan data:
% * for raw image and blurred images: use Features.generate2LMDiff to
% generate differences between 2 features in same channel
% * for all gradient and laplacian images, use Features.generate2LM to
% generate single features

dfs = {'type','','F',20,'radius',1,'nChn','REQ','pids',[],'randctr',false,...
  'neighbors',{},'fids',[]};
[type,F,radius,nChn,pids,randctr,neighbors,fids] = ...
  getPrmDflt(varargin,dfs,0);

assert(model.d==2,'Have not checked for 3D.');
assert(strcmp(type,'kborig_hack'));
assert(nChn==19,'Expected Jan data');

NCHAN_2PT = 3; % first 3 channels: raw, sig1=2, sig1=4
NCHAN_1PT = 16; % last 16 channels: gradient + laplacian images
xs2pt = Features.generate2LMDiff(model,'F',F,'radiusFac',radius,...
  'randctr',randctr,'neighbors',neighbors,'fids',fids,'nchan',NCHAN_2PT);

xs1pt = Features.generate2LM(model,'F',F,'radiusFac',radius,...
  'randctr',randctr,'neighbors',neighbors,'fids',fids,'nchan',NCHAN_1PT);
chans1 = xs1pt(:,6);
assert(all(ismember(chans1,1:NCHAN_1PT)));
xs1pt(:,6) = chans1+NCHAN_2PT; 

ftrData = struct('type',type,'F',F,'nChn',nChn,'xs2pt',xs2pt,'xs1pt',xs1pt);
end

%    'S:sig1=2.00'
%     'S:sig1=4.00'
%     'SGS:sig1=0.00,sig2=0.00'
%     'SGS:sig1=2.00,sig2=0.00'
%     'SGS:sig1=4.00,sig2=0.00'
%     'SGS:sig1=0.00,sig2=2.00'
%     'SGS:sig1=2.00,sig2=2.00'
%     'SGS:sig1=4.00,sig2=2.00'
%     'SGS:sig1=0.00,sig2=4.00'
%     'SGS:sig1=2.00,sig2=4.00'
%     'SGS:sig1=0.00,sig2=8.00'
%     'SLS:sig1=2.00,sig2=0.00'
%     'SLS:sig1=4.00,sig2=0.00'
%     'SLS:sig1=0.00,sig2=2.00'
%     'SLS:sig1=2.00,sig2=2.00'
%     'SLS:sig1=4.00,sig2=2.00'
%     'SLS:sig1=0.00,sig2=4.00'
%     'SLS:sig1=2.00,sig2=4.00'
        
function ftrData = ftrsGenIm( model, pStar, varargin )
% Generate random shape indexed features,
% relative to closest landmark (similar to Cao et al., CVPR12)
% Features are then computed using frtsCompIm
%
% USAGE
%  ftrData = ftrsGenIm( model, pStar, varargin )
%
% INPUTS
%  model    - shape model (see createModel())
%  pStar    - average shape (see initTr)
%  varargin - additional params (struct or name/value pairs)
%   .type     - [2] feature type (1 or 2)
%   .F        - [100] number of ftrs to generate
%   .radius   - [2] sample initial x from circle of given radius
%   .nChn     - [1] number of image channels (e.g. 3 for color images)
%   .pids     - [] part ids for each x
%
% OUTPUTS
%  ftrData  - struct containing generated ftrs
%   .type     - feature type (1 or 2)
%   .F        - total number of features
%   .nChn     - number of image channels
%   .xs       - feature locations relative to unit circle
%   .pids     - part ids for each x
%
% EXAMPLE
%
% See also shapeGt>ftrsCompIm

dfs={'type',2,'F',20,'radius',1,'nChn',3,'pids',[],'mask',[]};
[type,F,radius,nChn,pids,mask]=getPrmDflt(varargin,dfs,1);
%Generate random features on image
xs1=[];
while(size(xs1,1)<F),
    xs1=rand(F*1.5,2)*2-1;
    % xs1=xs1(xs1(:,1)<1&xs1(:,1)>-1&xs1(:,2)<1&xs1(:,2)>-1,:);
    xs1=xs1(sum(xs1.^2,2)<=1,:);
end
xs1=xs1(1:F,:);%*radius;

% KB not sure why it is like this
nfids = model.nfids;
% if(strcmp(model.name,'cofw') || strcmp(model.name,'fly_RF2'))
%     nfids=size(pStar,2)/3;
% else
%     nfids=size(pStar,2)/2;
% end

%Reproject each into closest pStar landmark
xs=zeros(F,3);%X,Y,landmark
for f=1:F
    posX=xs1(f,1)-pStar(1:nfids);
    posY=xs1(f,2)-pStar(nfids+1:nfids*2);
    dist = (posX.^2)+(posY.^2);
    [~,l]=min(dist);xs(f,:)=[posX(l) posY(l) l];
end
if(nChn>1),
    if(mod(type,2)==0),%make sure subbtractions occur inside same channel
        chns = randint2(F,1,[1 nChn]);
        xs(1:2:end,4) = chns; xs(2:2:end,4) = chns;
    else xs(:,4)=randint2(F,1,[1 nChn]);
    end
end
if(isempty(pids)), pids=floor(linspace(0,F,2)); end
ftrData=struct('type',type,'F',F,'nChn',nChn,'xs',xs,'pids',pids);
end

function [ftrs,occlD] = ftrsCompDup( model, phis, Is, ftrData,...
    imgIds, pStar, bboxes, occlPrm)
% Compute features from ftrsGenDup on Is 
%
% USAGE
%  [ftrs,Vs] = ftrsCompDup( model, phis, Is, ftrData, imgIds, pStar, ...
%       bboxes, occlPrm )
%
% INPUTS
%  model    - shape model
%  phis     - [MxR] relative shape for each image 
%  Is       - cell [N] input images [w x h x nChn] variable w, h
%  ftrData  - define ftrs to actually compute, output of ftrsGen
%  imgIds   - [Mx1] image id for each phi 
%  pStar   -  [1xR] average shape (see initTr) 
%  bboxes   - [Nx4] face bounding boxes 
%  occlPrm   - struct containing occlusion reasoning parameters
%    .nrows - [3] number of rows in face region
%    .ncols - [3] number of columns in face region
%    .nzones - [1] number of zones from where each regs draws
%    .Stot -  total number of regressors at each stage
%    .th - [0.5] occlusion threshold used during cascade
%
% OUTPUTS
%  ftrs     - [MxF] computed features
%  occlD    - struct containing occlusion info (if using full RCPR) 
%       .group    - [MxF] to which face region each features belong     
%       .featOccl - [MxF] amount of total occlusion in that area
%     
% EXAMPLE
%
% See also demoRCPR, shapeGt>ftrsGenDup
N = size(Is,1); nfids=model.nfids;
if(nargin<5 || isempty(imgIds)), imgIds=1:N; end
if(nargin<6 || isempty(pStar)),
  assert(false,'Unsupported compPhiStar sig.');
  %pStar=compPhiStar(model,phis,0,[]);
end
M=size(phis,1); assert(length(imgIds)==M);nChn=ftrData.nChn;

if(size(bboxes,1)==size(Is,1)), bboxes=bboxes(imgIds,:); end

assert(isnumeric(ftrData.type));
if(ftrData.type==3),
    FTot=ftrData.F;
    ftrs = zeros(M,FTot);
else
    FTot=ftrData.F;ftrs = zeros(M,FTot);
end
posrs = phis(:,nfids+1:nfids*2);poscs = phis(:,1:nfids);
useOccl=occlPrm.Stot>1;
if(useOccl && (strcmp(model.name,'cofw') || strcmp(model.name,'fly_RF2')))
    occl = phis(:,(nfids*2)+1:nfids*3);
    occlD=struct('featOccl',zeros(M,FTot),'group',zeros(M,FTot));
else occl = zeros(M,nfids);occlD=[];
end
%GET ALL POINTS
if(nargout>1)
    [csStar,rsStar]=getLinePoint(ftrData.xs,pStar(1:nfids),...
        pStar(nfids+1:nfids*2));
    pos=ftrsOcclMasks([csStar' rsStar']);
end
%relative to two points
[cs1,rs1]=getLinePoint(ftrData.xs,poscs,posrs); 
nGroups=occlPrm.nrows*occlPrm.ncols;
%ticId =ticStatus('Computing feats',1,1,1);
for n=1:M
    img = Is{imgIds(n),1}; [h,w,ch]=size(img); 
    if(ch==1 && nChn==3), img = cat(3,img,img,img);
    elseif(ch==3 && nChn==1), img = rgb2gray(img);
    end
    cs1(n,:)=max(1,min(w,cs1(n,:)));
    rs1(n,:)=max(1,min(h,rs1(n,:)));
    
    %where are the features relative to bbox?
    if(useOccl && (strcmp(model.name,'cofw') || strcmp(model.name,'fly_RF2')))
        %to which group (zone) does each feature belong?
        occlD.group(n,:)=codifyPos((cs1(n,:)-bboxes(n,1))./bboxes(n,3),...
            (rs1(n,:)-bboxes(n,2))./bboxes(n,4),...
            occlPrm.nrows,occlPrm.ncols);
        %to which group (zone) does each landmark belong?
        groupF=codifyPos((poscs(n,:)-bboxes(n,1))./bboxes(n,3),...
            (posrs(n,:)-bboxes(n,2))./bboxes(n,4),...
            occlPrm.nrows,occlPrm.ncols);
        %NEW
        %therefore, what is the occlusion in each group (zone)
        occlAm=zeros(1,nGroups);
        for g=1:nGroups
            occlAm(g)=sum(occl(n,groupF==g));
        end
        %feature occlusion = sum of occlusion on that area
        occlD.featOccl(n,:)=occlAm(occlD.group(n,:));
    end
    
    inds1 = (rs1(n,:)) + (cs1(n,:)-1)*h;
    if(nChn>1), inds1 = inds1+(chs'-1)*w*h; end
    
    if(isa(img,'uint8')), 
      ftrs1=double(img(inds1)')/255;
    elseif isa(img,'uint16'),
      ftrs1=double(img(inds1)')/(2^16-1);
    else ftrs1=double(img(inds1)'); end
    
    assert(isnumeric(ftrData.type));
    if ftrData.type==3
      ftrs1=ftrs1*2-1; 
      ftrs(n,:)=reshape(ftrs1,1,FTot);
    else
      ftrs(n,:)=ftrs1;
    end
    %tocStatus(ticId,n/M);
end
end

function ftrs = ftrsCompKBOrig(model,phis,Is,ftrData,imgIds,pStar,bboxes,occlPrm)

%ftrData = struct('type',type,'F',F,'nChn',nChn,'xs2pt',xs2pt,'xs1pt',xs1pt);

assert(ftrData.nChn==19);
% nChn==19 only used to assert against img size
ftrData2 = struct('type','2lmdiff','F',ftrData.F,'nChn',ftrData.nChn,'xs',ftrData.xs2pt);
ftrData1 = struct('type','2lm','F',ftrData.F,'nChn',ftrData.nChn,'xs',ftrData.xs1pt);

ftrs2 = ftrsCompDup2(model,phis,Is,ftrData2,imgIds,pStar,bboxes,occlPrm);
ftrs1 = ftrsCompDup2(model,phis,Is,ftrData1,imgIds,pStar,bboxes,occlPrm);
ftrs = [ftrs2 ftrs1];

end


%#3DOK
function [ftrs,occlD] = ftrsCompDup2( model, phis, Is, ftrData,...
    imgIds, pStar, bboxes, occlPrm, calrig)
% Compute features from ftrsGenDup2 on Is 
%
% INPUTS
%  model    - shape model
%  phis     - [MxD] shape for each image, absolute coords. 
% Is: struct with the following fields:
%   Is: vector of all N x nView images strung out in order of rows, pixels, channels, image, view
%   imszs: [2 x N x nView] size of each image
%   imoffs: [N x nView] offset for indexing image (view,i) (image will
%     be from off(view,i)+1:off(view,i)+imszs(1,view,i)*imszs(2,view,i)
%  ftrData  - scalar struct, feature definitions. output of ftrsGen
%  imgIds   - [Mx1] image id for each phi, indices into rows of Is.
%  pStar   -  [1xD] UNUSED ATM. average shape (see initTr)
%  bboxes   - [Nx4] UNUSED ATM. face bounding boxes. Only used for occlusion.
%  occlPrm   - struct containing occlusion reasoning parameters
%    .nrows - [3] number of rows in face region
%    .ncols - [3] number of columns in face region
%    .nzones - [1] number of zones from where each regs draws
%    .Stot -  total number of regressors at each stage
%    .th - [0.5] occlusion threshold used during cascade
%
% OUTPUTS
%  ftrs     - [MxF] computed features
%  occlD    - struct containing occlusion info (if using full RCPR) 
%       .group    - [MxF] to which face region each features belong     
%       .featOccl - [MxF] amount of total occlusion in that area
%     
% EXAMPLE
%
% See also demoRCPR, shapeGt>ftrsGenDup

[M,D] = size(phis);
assert(D==model.D);

iscellIs = iscell(Is);
if iscellIs,
  [N,nviews] = size(Is);
else
  [N,nviews] = size(Is.imoffs);
end
assert(nviews==model.nviews);
if nargin<5 || isempty(imgIds)
  imgIds = 1:N; 
end
assert(length(imgIds)==M);
nfids = model.nfids;
nChn = ftrData.nChn;

% if size(bboxes,1)==N
%   bboxes = bboxes(imgIds,:); 
% end

% Extract/compute posrs, poscs (landmark pixel coords on images) from phis
% posrs, poscs: [MxnfidsxnView]
if strcmp(model.name,'mouse_paw3D')
  assert(false,'AL');
    %Now using a single merged image containing each view. If using one
    %image per view, select which image here.
    % Code for old 3D which uses PCA
%     imsz=size(Is{1});
%     X=[phis mean(Prm3D.X4)*ones(size(phis,1),1)];
%     phis=X/Prm3D.C;
%     phis(:,2)=phis(:,2)+imsz(2)/2;
%     nfids=nfids*2;
%     posrs = phis(:,nfids+1:nfids*2);
%     poscs = phis(:,1:nfids);
  imsz = size(Is{1});
  posrs = zeros(M,2*nfids);
  poscs = zeros(M,2*nfids);
  %rotmat = rodrigues(Prm3D.om0);
  if nfids>1, assert(false); end; % Speed up computation for more than nfids.
  if nfids == 1
    X3d = phis';
    X3d_right = bsxfun(@plus,rotmat*X3d,Prm3D.T0);
    phis_left = project_points2(X3d,zeros(3,1),zeros(3,1),Prm3D.fc_left,Prm3D.cc_left,Prm3D.kc_left);
    phis_right = project_points2(X3d_right,zeros(3,1),zeros(3,1),Prm3D.fc_right,Prm3D.cc_right,Prm3D.kc_right);
    posrs = [phis_left(2,:)' phis_right(2,:)'];
    poscs = [phis_left(1,:)' phis_right(1,:)'+imsz(2)/2];
  else
    for ndx = 1:size(phis,1)
      X3d = [phis(ndx,1:nfids);phis(ndx,nfids+1:2*nfids);phis(ndx,2*nfids+1:3*nfids)];
      X3d_right = rotmat*X3d+Prm3D.T0;
      phis_left = project_points2(X3d,zeros(3,1),zeros(3,1),Prm3D.fc_left,Prm3D.cc_left,Prm3D.kc_left);
      phis_right = project_points2(X3d_right,zeros(3,1),zeros(3,1),Prm3D.fc_right,Prm3D.cc_right,Prm3D.kc_right);
      % posrs store y-coordinates and while poscs store x-cordinates. check
      % by debugging.
      posrs(ndx,:) = [phis_left(2,:) phis_right(2,:)];
      poscs(ndx,:) = [phis_left(1,:) phis_right(1,:)+imsz(2)/2];
    end
  end
  nfids = nfids*2;
elseif model.d==3
  % phis are 3D coords in coord sys of iViewBase
  iViewBase = model.Prm3D.iViewBase;
  
  Pbase = reshape(phis,M,nfids,model.d); % [Mxnfidsx3]
  Pbase = permute(Pbase,[3 1 2]); % [3xMxnfids]
  %Pbase = reshape(Pbase,3,M*nfids); % [3xM*nfids]. all pt1's, then all pt2's, ...

  crigAll = calrig;
  assert(isa(crigAll,'CalRig') && isvector(crigAll) && numel(crigAll)==N);
  crigUn = unique(crigAll);
  nCrigUn = numel(crigUn);
  crigP = crigAll(imgIds);
  
  % Compute projected r/c pixel locs for each shape/pt/vw
  posrs = nan(M,nfids,nviews);
  poscs = nan(M,nfids,nviews);

  for iCrigUn = 1:nCrigUn
    crig = crigUn(iCrigUn);
    tfP = crigP==crig;
    nPThisCrig = nnz(tfP);
    PbaseThisCrig = Pbase(:,tfP,:);
    PbaseThisCrig = reshape(PbaseThisCrig,[3 nPThisCrig*nfids]);

    for iView = 1:nviews
      if iView==iViewBase
        PviewThisCrig = PbaseThisCrig;
      else
        PviewThisCrig = crig.viewXformCPR(PbaseThisCrig,iViewBase,iView);
      end
      assert(isequal(size(PviewThisCrig),[3 nPThisCrig*nfids]));
      [rview,cview] = crig.projectCPR(PviewThisCrig,iView);
      % rview/cview are [nPThisCrig*nfids] col vecs; all pt1's, the all pt2's, ...
      posrs(tfP,:,iView) = reshape(rview,nPThisCrig,nfids);
      poscs(tfP,:,iView) = reshape(cview,nPThisCrig,nfids);
    end
  end  
elseif model.d==2 && nviews==1
  posrs = phis(:,nfids+1:D);
  poscs = phis(:,1:nfids);
else
  assert(false);
end
if nviews==1
  assert(isequal(size(posrs),size(poscs),[M,nfids]));
else
  assert(isequal(size(posrs),size(poscs),[M,nfids,nviews]));
end

FTot = ftrData.F;
assert(FTot==size(ftrData.xs,1));
useOccl = occlPrm.Stot>1;
if useOccl && (strcmp(model.name,'cofw') || strcmp(model.name,'fly_RF2'))
  assert(false,'AL');
  occl = phis(:,(nfids*2)+1:nfids*3);
  occlD = struct('featOccl',zeros(M,FTot),'group',zeros(M,FTot));
else
  occl = zeros(M,nfids);
  occlD = [];
end
%GET ALL POINTS
switch ftrData.type
  case 'single landmark'
    [cs1,rs1,vw] = Features.compute1LM(ftrData.xs,poscs,posrs);
    chn = ones(size(cs1));
  case '2lm'
    [cs1,rs1,chn,vw] = Features.compute2LM(ftrData.xs,poscs,posrs);
  case 'two landmark elliptical'
    [cs1,rs1,chn,vw] = Features.compute2LMelliptical(ftrData.xs,poscs,posrs);    
  case '2lmdiff'
    [cs1,rs1,cs2,rs2,chn,vw] = Features.compute2LMDiff(ftrData.xs,poscs,posrs);
  otherwise
    assert(false);
end

% nGroups = occlPrm.nrows*occlPrm.ncols;
%ticId =ticStatus('Computing feats',1,1,1);

ftrs = nan(M,FTot);
assert(isequal(size(cs1),size(rs1),size(ftrs)));
assert(isequal(size(vw),[FTot 1]));
assert(all(ismember(vw,1:nviews)));

% KB 20180419: vectorized version of commented code below

if iscellIs,
  chs = cellfun(@(x) size(x,3), Is);
  assert(all(chs(:)==nChn));
else
  % chs should all be nChn now, not saved
end

cs1 = max(cs1,1);
rs1 = max(rs1,1);
if strcmp(ftrData.type,'2lmdiff')
  cs2 = max(1,cs2);
  rs2 = max(1,rs2);
end

for iview = 1:nviews
  
  tfvw = vw==iview; % [F] logical. 1 where cols of cs1,rs1,ftrs are for current view  
  
  if iscellIs,
    hs = cellfun(@(x) size(x,1), Is(:,iview));
    ws = cellfun(@(x) size(x,2), Is(:,iview));
  else
    hs = Is.imszs(1,:,iview)';
    ws = Is.imszs(2,:,iview)';
  end
  hs = hs(imgIds);
  ws = ws(imgIds);
  
  hsun = unique(hs);
  wsun = unique(ws);
  fast = all(tfvw) && isscalar(hsun) && isscalar(wsun);  
  
  % old matlab requires explicit bsxfun
  if verLessThan('matlab','9.2.0'),
    cs1(:,tfvw) = bsxfun(@min,ws,cs1(:,tfvw));
    rs1(:,tfvw) = bsxfun(@min,hs,rs1(:,tfvw));
  elseif fast
    cs1 = min(wsun,cs1);
    rs1 = min(hsun,rs1);
  else
    cs1(:,tfvw) = min(ws,cs1(:,tfvw));
    rs1(:,tfvw) = min(hs,rs1(:,tfvw));
  end
    
  if (useOccl && (strcmp(model.name,'cofw') || strcmp(model.name,'fly_RF2')))
    assert(false,'AL');
  end

  % old matlab requires explicit bsxfun
  if verLessThan('matlab','9.2.0'),
    inds1s = rs1(:,tfvw) + bsxfun(@times,cs1(:,tfvw)-1,hs) + bsxfun(@times,chn(:,tfvw)-1,hs.*ws);
  elseif fast
    inds1s = rs1 + (cs1-1)*hsun + (chn-1)*(hsun*wsun);
  else
    inds1s = rs1(:,tfvw) + (cs1(:,tfvw)-1).*hs + (chn(:,tfvw)-1).*(hs.*ws);
  end
  
  if strcmp(ftrData.type,'2lmdiff')
    % old matlab requires explicit bsxfun
    if verLessThan('matlab','9.2.0'),
      cs2(:,tfvw) = bsxfun(@min,ws,cs2(:,tfvw));
      rs2(:,tfvw) = bsxfun(@min,hs,rs2(:,tfvw));
      inds2s = rs2(:,tfvw) + bsxfun(@times,cs2(:,tfvw)-1,h) + bsxfun(@times,chn(:,tfvw)-1,h.*w);
    else
      % TODO optim fast
      cs2(:,tfvw) = min(ws,cs2(:,tfvw));
      rs2(:,tfvw) = min(hs,rs2(:,tfvw));
      inds2s = rs2(:,tfvw) + (cs2(:,tfvw)-1).*h + (chn(:,tfvw)-1).*h.*w;
    end
  end
    
  switch ftrData.type
    case {'single landmark' '2lm' 'two landmark elliptical'}
      
      if iscellIs,
        for n = 1:M,
          ftrs(n,tfvw) = hlpFtr(Is{imgIds(n),iview},inds1s(n,:));
        end
      else
        if verLessThan('matlab','9.2.0'),
          ftrs(:,tfvw) = Is.Is(bsxfun(@plus,inds1s,Is.imoffs(imgIds,iview)));
        elseif fast
          ftrs = Is.Is(inds1s+Is.imoffs(imgIds,iview));
          szassert(ftrs,[M FTot]);
        else
          ftrs(:,tfvw) = Is.Is(inds1s+Is.imoffs(imgIds,iview));
        end
      end
              
    case '2lmdiff'

      if iscellIs,
        for n = 1:M,
          ftrs1 = hlpFtr(Is{imgIds(n),iview},inds1s(n,:));
          ftrs2 = hlpFtr(Is{imgIds(n),iview},inds2s(n,:));
          ftrs(n,tfvw) = ftrs1-ftrs2;
        end
      else
      
        if verLessThan('matlab','9.2.0'),
          ftrs1 = Is.Is(bsxfun(@plus,inds1s,Is.imoffs(imgIds,iview)));
          ftrs2 = Is.Is(bsxfun(@plus,inds2s,Is.imoffs(imgIds,iview)));
        else
          ftrs1 = Is.Is(inds1s+Is.imoffs(imgIds,iview));
          ftrs2 = Is.Is(inds2s+Is.imoffs(imgIds,iview));
        end
        ftrs(:,tfvw) = ftrs1-ftrs2;
      end
         
  end
end

% for iview = 1:nviews
%   
%   tfvw = vw==iview; % [F] logical. 1 where cols of cs1,rs1,ftrs are for current view  
%   for n = 1:M
%     
%     % Crop feature positions cs1,rs1 to image. This is weird, hope this 
%     % doesn't happen too often.
%     img = Is{imgIds(n),iview};
%     [h,w,ch] = size(img);
%     assert(nChn==ch);
%     
%     cs1(n,tfvw) = max(1,min(w,cs1(n,tfvw)));
%     rs1(n,tfvw) = max(1,min(h,rs1(n,tfvw)));
%     if strcmp(ftrData.type,'2lmdiff')
%       cs2(n,tfvw) = max(1,min(w,cs2(n,tfvw)));
%       rs2(n,tfvw) = max(1,min(h,rs2(n,tfvw)));
%     end
%     
%     %where are the features relative to bbox?
%     if (useOccl && (strcmp(model.name,'cofw') || strcmp(model.name,'fly_RF2')))
%       assert(false,'AL');
%       %         %to which group (zone) does each feature belong?
%       %         occlD.group(n,:)=codifyPos((cs1(n,:)-bboxes(n,1))./bboxes(n,3),...
%       %             (rs1(n,:)-bboxes(n,2))./bboxes(n,4),...
%       %             occlPrm.nrows,occlPrm.ncols);
%       %         %to which group (zone) does each landmark belong?
%       %         groupF=codifyPos((poscs(n,:)-bboxes(n,1))./bboxes(n,3),...
%       %             (posrs(n,:)-bboxes(n,2))./bboxes(n,4),...
%       %             occlPrm.nrows,occlPrm.ncols);
%       %         %NEW
%       %         %therefore, what is the occlusion in each group (zone)
%       %         occlAm=zeros(1,nGroups);
%       %         for g=1:nGroups
%       %             occlAm(g)=sum(occl(n,groupF==g));
%       %         end
%       %         %feature occlusion = sum of occlusion on that area
%       %         occlD.featOccl(n,:)=occlAm(occlD.group(n,:));
%     end
%     
%     inds1 = rs1(n,tfvw) + (cs1(n,tfvw)-1)*h + (chn(n,tfvw)-1)*h*w;
%     ftrs1 = hlpFtr(img,inds1);
%     switch ftrData.type
%       case {'single landmark' '2lm' 'two landmark elliptical'}
%         ftrs(n,tfvw) = ftrs1;
%       case '2lmdiff'
%         inds2 = rs2(n,tfvw) + (cs2(n,tfvw)-1)*h + (chn(n,tfvw)-1)*h*w;
%         ftrs2 = hlpFtr(img,inds2);
%         ftrs(n,tfvw) = ftrs1-ftrs2;
%       otherwise
%         assert(false);
%     end
%   end
% end

end

function ftrs1 = hlpFtr(img,inds1)
if isa(img,'uint8')
  ftrs1 = double(img(inds1)')/255;
elseif isa(img,'uint16')
  ftrs1 = double(img(inds1)')/(2^16-1);
else
  ftrs1 = img(inds1)';
end
end

function group=codifyPos(x,y,nrows,ncols)
%codify position of features into regions
nr=1/nrows;nc=1/ncols;
%Readjust positions so that they falls in [0,1]
x=min(1,max(0,x));y=min(1,max(0,y)); 
y2=y;x2=x;
for c=1:ncols,
    if(c==1), x2(x<=nc)=1; 
    elseif(c==ncols), x2(x>=nc*(c-1))=ncols;
    else x2(x>nc*(c-1) & x<=nc*c)=c;
    end
end
for r=1:nrows,
    if(r==1), y2(y<=nr)=1; 
    elseif(r==nrows), y2(y>=nc*(r-1))=nrows;
    else y2(y>nr*(r-1) & y<=nr*r)=r;
    end 
end
group=sub2ind2([nrows ncols],[y2' x2']);
end

function [cs1,rs1]=getLinePoint(FDxs,poscs,posrs)
%get pixel positions given coordinates as points in a line between
%landmarks
%INPUT NxF, OUTPUT NxF
if(size(poscs,1)==1)%pStar normalized
    l1= FDxs(:,1);l2= FDxs(:,2);xs=FDxs(:,3);
    x1 = poscs(:,l1);y1 = posrs(:,l1);
    x2 = poscs(:,l2);y2 = posrs(:,l2);
    
    a=(y2-y1)./(x2-x1); b=y1-(a.*x1);
    distX=(x2-x1)/2; ctrX= x1+distX;
    cs1=ctrX+(repmat(xs',size(distX,1),1).*distX);
    rs1=(a.*cs1)+b;
else
    if(size(FDxs,2)<4)%POINTS IN A LINE (ftrsGenDup)
        %2 points in a line with respect to center
        l1= FDxs(:,1);l2= FDxs(:,2);xs=FDxs(:,3);
        %center
        muX = mean(poscs,2);
        muY = mean(posrs,2);
        poscs=poscs-repmat(muX,1,size(poscs,2));
        posrs=posrs-repmat(muY,1,size(poscs,2));
        %landmark
        x1 = poscs(:,l1);y1 = posrs(:,l1);
        x2 = poscs(:,l2);y2 = posrs(:,l2);
        
        a=(y2-y1)./(x2-x1); b=y1-(a.*x1);
        distX=(x2-x1)/2; ctrX= x1+distX;
        cs1=ctrX+(repmat(xs',size(distX,1),1).*distX);
        rs1=(a.*cs1)+b;
        cs1=round(cs1+repmat(muX,1,size(FDxs,1)));
        rs1=round(rs1+repmat(muY,1,size(FDxs,1)));
    end
end
end

function [ftrs,occlD] = ftrsCompIm( model, phis, Is, ftrData,...
    imgIds, pStar, bboxes, occlPrm )
% Compute features from ftrsGenIm on Is 
%
% USAGE
%  [ftrs,Vs] = ftrsCompIm( model, phis, Is, ftrData, [imgIds] )
%
% INPUTS
%  model    - shape model
%  phis     - [MxR] relative shape for each image 
%  Is       - cell [N] input images [w x h x nChn] variable w, h
%  ftrData  - define ftrs to actually compute, output of ftrsGen
%  imgIds   - [Mx1] image id for each phi 
%  pStar   -  [1xR] average shape (see initTr) 
%  bboxes   - [Nx4] face bounding boxes 
%  occlPrm   - struct containing occlusion reasoning parameters
%    .nrows - [3] number of rows in face region
%    .ncols - [3] number of columns in face region
%    .nzones - [1] number of zones from where each regs draws
%    .Stot -  total number of regressors at each stage
%    .th - [0.5] occlusion threshold used during cascade
%
% OUTPUTS
%  ftrs     - [MxF] computed features
%  occlD    - [] empty structure
%
% EXAMPLE
%
% See also demoCPR, shapeGt>ftrsGenIm, shapeGt>ftrsCompDup

N = size(Is,1); nChn=ftrData.nChn;
if(nargin<5 || isempty(imgIds)), imgIds=1:N; end
M=size(phis,1); assert(length(imgIds)==M);

[pStar,phisN,bboxes] = ...
    compPhiStar( model, phis, 10, bboxes );

if(size(bboxes,1)==size(Is,1)), bboxes=bboxes(imgIds,:); end

F=size(ftrData.xs,1);ftrs = zeros(M,F);
useOccl=occlPrm.Stot>1;
if(strcmp(model.name,'cofw') || strcmp(model.name,'fly_RF2'))
    nfids=size(phis,2)/3;occlD=[];
else
    nfids=size(phis,2)/2;occlD=[];
end

%X,Y,landmark,Channel
rs = ftrData.xs(:,2);cs = ftrData.xs(:,1);xss = [cs';rs'];
ls = ftrData.xs(:,3);if(nChn>1),chs = ftrData.xs(:,4);end
%Actual phis positions
poscs=phis(:,1:nfids);posrs=phis(:,nfids+1:nfids*2);
%get positions of key landmarks
posrs=posrs(:,ls);poscs=poscs(:,ls);
%Reference points
X=[pStar(1:nfids);pStar(nfids+1:nfids*2)];
for n=1:M
    img = Is{imgIds(n)}; [h,w,ch]=size(img);
    if(ch==1 && nChn==3), img = cat(3,img,img,img);
    elseif(ch==3 && nChn==1), img = rgb2gray(img);
    end
    
    %Compute relation between phisN and pStar (scale, rotation)
    Y=[phisN(n,1:nfids);phisN(n,nfids+1:nfids*2)];
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    [~,~,Sc,Rot] = translate_scale_rotate(Y,X);
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %Compute feature locations by reprojecting
    aux=Sc*Rot*xss;
    %Resize accordingly to bbox size
    szX=bboxes(n,3)/2;szY=bboxes(n,4)/2;
    aux = [(aux(1,:)*szX);(aux(2,:)*szY)];
    
    %Add to respective landmark
    rs1 = round(posrs(n,:)+aux(2,:));
    cs1 = round(poscs(n,:)+aux(1,:));
    
    cs1 = max(1,min(w,cs1)); rs1=max(1,min(h,rs1));
    
    inds1 = (rs1) + (cs1-1)*h;
    if(nChn>1), chs = repmat(chs,1,m); inds1 = inds1+(chs-1)*w*h; end
    
    if(isa(img,'uint8')), 
      ftrs1=double(img(inds1)')/255;
    elseif isa(img,'uint16'),
      ftrs1=double(img(inds1)')/(2^16-1);
    else ftrs1=double(img(inds1)'); end
    
    assert(isnumeric(ftrData.type));
    if ftrData.type==1
        ftrs1=ftrs1*2-1; ftrs(n,:)=reshape(ftrs1,1,F);
    else ftrs(n,:)=ftrs1;
    end
end
end

function [h,w]=getSzIm(Is)
%get image sizes
N=size(Is,1); w=zeros(1,N);h=zeros(1,N);
for i=1:N, [w(i),h(i),~]=size(Is{i}); end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function phis = compose( model, phis0, phis1, bboxes )
% Compose two shapes phis0 and phis1: phis = phis0 + phis1.
%
% phis0: Shape, normalized
% phis1: Shape, absolute
%
% phis: Shape, normalized
%
% #ALOK
phis1 = projectPose(model,phis1,bboxes);
phis = phis0 + phis1;
end

function phis = inverse( model, phis0, bboxes )
% Compute inverse of two shapes phis0 so that phis0+phis=phis+phis0=identity.
%
% phis0: Shape, absolute
%
% phis: Shape, normalized
%
% #ALOK
phis = -projectPose(model,phis0,bboxes);
end

%#3DOK
function [phiNStar,phisN,bboxes] = compPhiStar(model,phis,pad,bboxes)
% Compute average location of points relative to box centers in normalized 
% coordinates
%
% phis: [MxD], shapes, absolute coords
% pad: scalar, number of pixels. Used when bboxes is not supplied; pad the 
% landmarks on each side by this amount to generate bboxes. CURRENTLY
% BBOXES MUST BE SUPPLIED SO PAD IS NEVER USED 
% bboxes (input): [Mx2*d].
%
% phiNStar: [1xD] phi (normalized coords) that minimizes sum of distances 
%   to phis, ie centroid
% phisN: [MxD] same as phis, but normalized to bboxes. All coords in range
% [-1,1] which maps to eg left-of-bbox, right-of-bbox.
% bboxes (output): [Mx2*d]. If bboxes is not supplied, they are generated 
% by taking the extent of each phi and padding with pad.

% nfids = model.nfids;
[M,D] = size(phis);
assert(D==model.D);
assert(isscalar(pad));
tfBboxesSupplied = exist('bboxes','var')>0;
assert(tfBboxesSupplied);
% if ~tfBboxesSupplied
%   if strcmp(model.name,'mouse_paw3D')
%     assert(false,'AL');
%     %       %left top width height
%     %       bboxes(n,1:3)=[min(phis(n,1:nfids))-pad ...
%     %         min(phis(n,nfids+1:2*nfids))-pad,...
%     %         min(phis(n,nfids*2+1:3*nfids))-pad];
%     %       bboxes(n,4)=max(phis(n,1:nfids))-bboxes(n,1)+2*pad;
%     %       bboxes(n,5)=max(phis(n,nfids+1:nfids*2))-bboxes(n,2)+2*pad;
%     %       bboxes(n,6)=max(phis(n,2*nfids+1:nfids*3))-bboxes(n,3)+2*pad;
%     %     szX=bboxes(imgIds(n),4)/2;szY=bboxes(imgIds(n),5)/2;szZ=bboxes(imgIds(n),6)/2;
%     %       ctr(1)=bboxes(imgIds(n),1)+szX;ctr(2) = bboxes(imgIds(n),2)+szY;ctr(3) = bboxes(imgIds(n),3)+szZ;
%   elseif model.d==2
%     %left top width height
%     % AL: minor bug here previously (height/width too big by pad) I think
%     xs = phis(:,1:nfids);
%     ys = phis(:,nfids+1:2*nfids);
%     xmin = min(xs,[],2);
%     xmax = max(xs,[],2);
%     ymin = min(ys,[],2);
%     ymax = max(ys,[],2);
%     bboxes = [xmin-pad ymin-pad xmax-xmin+2*pad ymax-ymin+2*pad];
%   else
%     assert(false,'AL');
%   end
% end
assert(isequal(size(bboxes),[M model.d*2]));
 
switch model.name
  case {'cofw' 'fly_RF2' 'mouse_paw3D'}
    assert(false,'Purely historical, prob works fine.');
  otherwise
    % none
end
    
phisN = projectPose(model,phis,bboxes);
phiNStar = mean(phisN,1);
end

%#3DOK
function phis1 = reprojectPose(model,phisN,bboxes)
% Reproject normalized shape onto bounding boxes
%
% phisN: [MxD] Normalized shapes (all coords in range [-1,1] for shapes
%   fully enclosed by bboxes)
% bboxes: [Mx2*d] bounding boxes
% 
% phis1: [MxD] Shapes in absolute (non-normalized) coords, relative to
%   bboxes

[nOOB,tfOOB] = Shape.normShapeOOB(phisN);
if nOOB>0
  warningNoTrace('reprojPose: %d coords out-of-bounds. Clipping...',nOOB);
end
% Note phisN(tfOOB) will have no nans; otherwise would want to use
% 'includenan' option for max/min
phisN(tfOOB) = max(min(phisN(tfOOB),1),-1); 

[M,D] = size(phisN);
assert(D==model.D);
assert(size(bboxes,1)==M);
d = size(bboxes,2)/2;
assert(d==model.d);

nfids = model.nfids;
radii = bboxes(:,d+1:end)/2; % Mxd
ctrs = bboxes(:,1:d) + radii; % Mxd

phisN = reshape(phisN,[M,nfids,d]);
phis1 = bsxfun(@plus,bsxfun(@times,phisN,reshape(radii,[M,1,d])),...
                     reshape(ctrs,[M,1,d]));
phis1 = reshape(phis1,[M,nfids*d]);

% if  (strcmp(model.name,'mouse_paw3D'))
%     szX=bboxes(:,4)/2;szY=bboxes(:,5)/2;szZ=bboxes(:,6)/2;
%     ctrX = bboxes(:,1)+szX;ctrY = bboxes(:,2)+szY;ctrZ = bboxes(:,3)+szZ;
%     szX=repmat(szX,1,nfids);szY=repmat(szY,1,nfids);szZ=repmat(szZ,1,nfids);
%     ctrX=repmat(ctrX,1,nfids);ctrY=repmat(ctrY,1,nfids);ctrZ=repmat(ctrZ,1,nfids);
% else
%     szX=bboxes(:,3)/2;szY=bboxes(:,4)/2;
%     ctrX = bboxes(:,1)+szX;ctrY = bboxes(:,2)+szY;
%     szX=repmat(szX,1,nfids);szY=repmat(szY,1,nfids);
%     ctrX=repmat(ctrX,1,nfids);ctrY=repmat(ctrY,1,nfids);
% end
% if(strcmp(model.name,'cofw') || strcmp(model.name,'fly_RF2'))
%     phis1 = [(phis(:,1:nfids).*szX)+ctrX (phis(:,nfids+1:nfids*2).*szY)+ctrY...
%         phis(:,(nfids*2)+1:nfids*3)];
% elseif (strcmp(model.name,'mouse_paw3D'))
%     phis1 = [(phis(:,1:nfids).*szX)+ctrX (phis(:,nfids+1:nfids*2).*szY)+ctrY (phis(:,2*nfids+1:nfids*3).*szZ)+ctrZ];
% else
%     phis1 = [(phis(:,1:nfids).*szX)+ctrX (phis(:,nfids+1:nfids*2).*szY)+ctrY];
% end

end

%# 3DOK
function phisN = projectPose(model,phis,bboxes)
% Project shape into normalized coords given bounding boxes
% phis: [MxD] Shapes in absolute coords
% bboxes: [Mx2*d] bounding boxes. Format of each row is 
% [offsetX offsetY widthX widthY] or 
% [offsetX offsetY offsetZ widthX widthY widthZ]
% 
% phisN: [MxD] Shapes in normalized coords relative to bboxes (coords in
% (-1,1) if bboxes fully enclose all shapes

[M,D] = size(phis);
assert(D==model.D);

assert(size(bboxes,1)==M);
d = size(bboxes,2)/2; % for each dimension, two vals: coord (offset) and width
assert(d==model.d);

nfids = model.nfids;
radii = bboxes(:,d+1:end)/2; % Mxd
ctrs = bboxes(:,1:d) + radii; % Mxd

phis = reshape(phis,[M,nfids,d]);
phisN = bsxfun(@rdivide,bsxfun(@minus,phis,reshape(ctrs,[M,1,d])),...
                        reshape(radii,[M,1,d]));
phisN = reshape(phisN,[M,nfids*d]);

% if  (strcmp(model.name,'mouse_paw3D'))
%     szX=bboxes(:,4)/2;szY=bboxes(:,5)/2;szZ=bboxes(:,6)/2;
%     ctrX=bboxes(:,1)+szX;ctrY=bboxes(:,2)+szY;ctrZ=bboxes(:,3)+szZ;
%     szX=repmat(szX,1,nfids);szY=repmat(szY,1,nfids);szZ=repmat(szZ,1,nfids);
%     ctrX=repmat(ctrX,1,nfids);ctrY=repmat(ctrY,1,nfids);ctrZ=repmat(ctrZ,1,nfids);
% else
%     szX=bboxes(:,3)/2;szY=bboxes(:,4)/2;
%     ctrX=bboxes(:,1)+szX;ctrY=bboxes(:,2)+szY;
%     szX=repmat(szX,1,nfids);szY=repmat(szY,1,nfids);
%     ctrX=repmat(ctrX,1,nfids);ctrY=repmat(ctrY,1,nfids);
% end
% if(strcmp(model.name,'cofw') || strcmp(model.name,'fly_RF2'))
%     phis = [(phis(:,1:nfids)-ctrX)./szX (phis(:,nfids+1:nfids*2)-ctrY)./szY ...
%         phis(:,(nfids*2)+1:nfids*3)];
% elseif (strcmp(model.name,'mouse_paw3D'))
%     phis = [(phis(:,1:nfids)-ctrX)./szX (phis(:,nfids+1:nfids*2)-ctrY)./szY (phis(:,2*nfids+1:nfids*3)-ctrZ)./szZ];
% else  phis = [(phis(:,1:nfids)-ctrX)./szX (phis(:,nfids+1:nfids*2)-ctrY)./szY];
% end
end

%# 3DOK
function [ds,dsAll] = dist(model,phis0,phis1)
% Compute distance between two sets of shapes.
%
% phis0: [NxDxT]
% phis1: [NxDxT] or [NxD]
%
% dsAll: [NxnfidsxT]. 2-norm distances between phis0 and phis1
% ds: [Nx1xT]. Equal to mean(dsAll,2).

[N,D,T] = size(phis0); 
szphis1 = size(phis1);
assert(isequal(szphis1,[N D T]) || isequal(szphis1,[N D]));

del = bsxfun(@minus,phis0,phis1);
nfids = model.nfids;
d = D/nfids;

switch model.name
  case {'lfpw' 'cofw' 'lfw' 'helen' 'pie' 'apf'}
    assert(false);
  otherwise
    distPup = 1;
end
% %Distance between pupils
% if(strcmp(model.name,'lfpw') || strcmp(model.name,'cofw'))
%     distPup=sqrt(((phis1(:,17)-phis1(:,18)).^2) + ...
%         ((phis1(:,17+nfids)-phis1(:,18+nfids)).^2));
%     distPup = repmat(distPup,[1,nfids,T]);
% elseif(strcmp(model.name,'lfw'))
%     leyeX=mean(phis1(:,1:2),2);leyeY=mean(phis1(:,(1:2)+nfids),2);
%     reyeX=mean(phis1(:,7:8),2);reyeY=mean(phis1(:,(7:8)+nfids),2);
%     distPup=sqrt(((leyeX-reyeX).^2) + ((leyeY-reyeY).^2));
%     distPup = repmat(distPup,[1,nfids,T]);
% elseif(strcmp(model.name,'helen'))
%     leye = [mean(phis1(:,135:154),2) mean(phis1(:,nfids+(135:154)),2)];
%     reye = [mean(phis1(:,115:134),2) mean(phis1(:,nfids+(115:134)),2)];
%     distPup=sqrt(((reye(:,1)-leye(:,1)).^2)+...
%         ((reye(:,2)-leye(:,2)).^2));
%     distPup = repmat(distPup,[1,nfids,T]);
% elseif(strcmp(model.name,'pie'))
%     leye = [mean(phis1(:,37:42),2) mean(phis1(:,nfids+(37:42)),2)];
%     reye = [mean(phis1(:,43:48),2) mean(phis1(:,nfids+(43:48)),2)];
%     distPup=sqrt(((reye(:,1)-leye(:,1)).^2)+...
%         ((reye(:,2)-leye(:,2)).^2));
%     distPup = repmat(distPup,[1,nfids,T]);
% elseif(strcmp(model.name,'apf'))
%     leye = [mean(phis1(:,7:8),2) mean(phis1(:,nfids+(7:8)),2)];
%     reye = [mean(phis1(:,9:10),2) mean(phis1(:,nfids+(9:10)),2)];
%     distPup=sqrt(((reye(:,1)-leye(:,1)).^2)+...
%         ((reye(:,2)-leye(:,2)).^2));
% else
% %   if (strcmp(model.name,'larva')) || (strcmp(model.name,'mouse_paw')) ||...
% %         (strcmp(model.name,'mouse_paw2')) || (strcmp(model.name,'mouse_paw3D')) ||...
% %         (strcmp(model.name,'fly_RF2'))||(strcmp(model.name,'mouse_paw_multi'))
%     %distPup=ones(size(phis1,1),nfids,T);
%     distPup = 1;
% end

sz = size(del);
assert(numel(sz)<=3);
% if numel(sz) < 4,
%   sz = [sz,ones(1,4-numel(sz))];
% end
deltmp = reshape(del,[N nfids d T]);
dsAll = sqrt(sum(deltmp.^2,3)); % [Nxnfidsx1xT]
dsAll = reshape(dsAll,[N nfids T]);
%dsAll = reshape(dsAll,[N,nfids,prod(sz(3:end))])./distPup;
ds = mean(dsAll,2);

if true
  if d==3
    dsAll0 = sqrt((del(:,1:nfids,:).^2) + (del(:,nfids+1:nfids*2,:).^2) + (del(:,2*nfids+1:nfids*3,:).^2));
  elseif d==2
    dsAll0 = sqrt(del(:,1:nfids,:).^2 + del(:,nfids+1:nfids*2,:).^2); % [NxnfidsxT]
  else
    error('d should be 2 or 3');
  end
  dsAll0 = dsAll0./distPup; 
  assert(isequaln(dsAll,dsAll0));
  %ds0=mean(dsAll0,2);%2*sum(dsAll,2)/R;
end

end

function [pCur,pGt,pGtN] = initTr(pGt,model,bboxes,L,dorotate)
assert(false,'Deprecated, use Shape.randInitShapes.');
end

function p = initTest(bboxes,model,pGtN,RT1,dorotate,varargin)
assert(false,'Deprecated, use Shape.randInitShapes.');
end

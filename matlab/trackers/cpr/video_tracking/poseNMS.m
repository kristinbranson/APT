function Y = poseNMS( X, S, R, maxK,  prmTrack )
%% "Merging Pose Estimates Across Space and Time". X.P. Burgos-Artizzu,
% D.Hall, P.Perona, P.Dollar. BMVC 2013, Bristol, UK.
%
% Compute sequence of trajectories explaining observed detections for a 
% unknown number of objects (Section 2.3 in paper). 
% (For known number of objects see poseNMS_K.m)
%
%
% USAGE
%  Y = poseNMS( X, S, R, maxK, varargin )
%
% INPUTS
%  X          - [1xT] cell of [ntxp] centers 
%                    nt=number of detections in frame t
%                    p=detection/pose parametrization
%  S          - [1xT] cell of [ntx1] scores
%  R          - [1xT] cell of [] or [ntxm] candidate matrices
%                   (Appendix B in paper)
%                    m=number of distinct objects
%                    R says to which object detections nt belong (0/1)
%  maxK       - maximum number of objects that can be present 
%                (infinite if no boundary)
%  varargin   - additional params (struct or name/value pairs)
%   .Y          - [] initial solution (must be mxpxT)
%   .norm       - [1] distance normalization
%   .th         - [1] distance supression threshold
%   .isAng      - [0] length p vector indicating dims that are angles
%   .lambda     - [.1] time weighting (constant position)
%   .lambda2    - [0] weighting of smoothness term
%   .nPhase     - [4] number of optimization phases
%   .window     - [50] window size
%   .symmetric  - [1] (simmetric=1, backward only=0) window
%   .show       - [0] figure to display results in
%   .bnds       - [] parse video into k segments: bnds(i)+1:bnds(i+1)
%   .ms         - [] parse video into k segments with m=ms(i) per segm.
%
% OUTPUTS
%  Y          - [mxpxT] tracking solution
%
% EXAMPLE
%
% See also poseNMS_K
%
% Copyright 2013 X.P. Burgos-Artizzu, D.Hall, P.Perona and Piotr Dollar.  
%  [xpburgos-at-gmail-dot-com]
% Please email me if you find bugs, or have suggestions or questions!
% Licensed under the Simplified BSD License [see bsd.txt]

%copy inputs
bnds=prmTrack.bnds;ms=prmTrack.ms;
%If no bounds specified, perform tracking on full video
if(isempty(bnds))
    Y=poseNMSBatch(X, S, R, maxK,  prmTrack );
else
    %Ow, call with different bounds
    prmTrack.bnds=[];prmTrack.ms=[];
    Y=zeros(max(ms),size(X{1},2),length(X));
    for k=1:length(ms)
        batch=bnds(k)+1:bnds(k+1);
        aux=poseNMSBatch(X(batch),S(batch),R(batch),ms(k),prmTrack);
        Y(1:size(aux,1),:,batch)=aux;
    end
end
end

function Y=poseNMSBatch(X, S, R, maxK,  prmTrack)
z=prmTrack.norm*prmTrack.norm;
Y=[];k=1;ndet=1e5;RA=cell2array(R);
if(~isempty(RA)), maxK=max(maxK,size(RA,2)); end
%Call tracker once
while(ndet>=prmTrack.th && k<=maxK)
    %If no Identity info, call with all detections
    if(isempty(RA)),
        Y1 = poseNMS_K(X,S,R,1,prmTrack);
        %Supress nearby scores
        [S,ndet] = supress(X,Y1,S,z,prmTrack.isAng);
        %If identity info, call one detected object at a time, using R
    else
        T=length(S); X1=cell(1,T);R1=cell(1,T);S1=cell(1,T);%S1=S;
        for t=1:T,
            if(~isempty(R{t})),
                ind=find(R{t}(:,k)==1);
                S1{t}=S{t}(ind);
                X1{t}=X{t}(ind,:);
                %S1{t}(R{t}(:,k)==0)=0;
            end
        end
        Y1=poseNMS_K(X1,S1,R1,1,prmTrack);
        %Supress nearby scores
        [S,ndet] = supress(X,Y1,S,z,prmTrack.isAng);
    end
    %copy trajectory into Y
    Y(k,:,:)=Y1;k=k+1;
end
end

%Supress scores S of detections X near path Y
function [S2,nleft]=supress(X,Y,S,z,isAng)
T=length(X);S2=S;nleft=0;

% if any angle, convert X/Y from standard to extended format
isAng=logical(isAng);pAng=sum(isAng);
if(pAng>0), 
  [~,ord]=sort(isAng);
  for i=1:T, X{i}=convToExt(X{i}(:,ord),pAng); end
  if(~isempty(Y)), Y=convToExt(Y(:,ord,:),pAng); end
end

for t=1:T
    sc_t=S2{t}; 
    %compute distance from pred to detections
    %all those inside z are set to 0
    %the rest are untouched
    if(~isempty(X{t})), D=dist(X{t},Y(1,:,t)); sc_t(D<z)=0; end
    S2{t}=sc_t;nleft=nleft+(length(find(sc_t>0)));
end
nleft=nleft/T;
end

function D = dist(X,Y)
% All pairs thresholded squared euclidean distance.
%  X - [n x p] matrix of n p-dim vectors
%  Y - [m x p] matrix of m p-dim vectors
%  R - [n x m] if given, R(i,j)=0 implies D(i,j)=1
%  z - [1 x 1] max distance
%  D - [n x m] distance matrix
Yt=Y'; XX=sum(X.*X,2); YY=sum(Yt.*Yt,1);
D=bsxfun(@plus,XX,YY)-2*X*Yt; D=D./size(X,2);
end

function Y = convToExt( Y, pAng )
% convert from standard format (a,b,ang) to extended format (a,b,x,y)
if(pAng==0), return; end; p0=size(Y,2)-pAng;
Y=[Y(:,1:p0,:) cos(Y(:,p0+1:end,:)) sin(Y(:,p0+1:end,:))];
end
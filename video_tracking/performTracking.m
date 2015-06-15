function Y=performTracking(pRT,bbs,good,th)
%% MERGE ESTIMATES
[T,D,RT1]=size(pRT);X=cell(1,T);S=cell(1,T);R=cell(1,T);
for t=1:T
    if(good(t))
        X{t}= permute(pRT(t,:,:),[3 2 1]);
        S{t}= repmat(mean(bbs{t}(:,5)),RT1,1);
    end
end
%%Track positions
prmTrack=struct('norm',100,'th',1,'lambda',.25,'lambda2',0,...
  'nPhase',4,'window',1000,'symmetric',1,'isAng',zeros(1,D),...
  'ms',[],'bnds',[]); 
Y = poseNMS( X, S, R, 1, prmTrack );
Y=permute(Y,[3 2 1]);
%If using COFW model, binarize occlusion according to learnt threshold
if(th~=-1)
    occl=Y(:,59:end); 
    occl(occl<th)=0; occl(occl>=th)=1;
    Y(:,59:end)=occl; 
end
end
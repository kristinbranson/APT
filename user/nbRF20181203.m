%%
%ld
%t

vcd = ld.viewCalibrationData;
cr = vcd{1};

nphyspts = 19;
nviews = 2;
n = height(t);
pLbl = reshape(t.p,[n nphyspts nviews 2]); % [i,ipt,ivw,x/y]

tic;
pcam1 = permute(pLbl(:,:,1,:),[4 1 2 3]); % 2 x n x nphyspts
pcam2 = permute(pLbl(:,:,2,:),[4 1 2 3]);
pcam1 = reshape(pcam1,2,n*nphyspts);
pcam2 = reshape(pcam2,2,n*nphyspts);
rperr1 = nan(n*nphyspts,1);
rperr2 = nan(n*nphyspts,1);
wbObj = WaitBarWithCancelCmdline('strotri');
wbObj.startPeriod('strotri','shownumden',true,'denominator',n*nphyspts);
for i=1:n*nphyspts
  wbObj.updateFracWithNumDen(i);
  [~,~,~,rperr1(i),rperr2(i)] = cr.stereoTriangulateML(pcam1(:,i),pcam2(:,i));
end

toc

rperr1 = reshape(rperr1,n,nphyspts);
rperr2 = reshape(rperr2,n,nphyspts);

CalRigMLStro.rperrPlot(rperr1,rperr2);

tstr = sprintf('RP err (nLblRows=%d)',n);
title(tstr,'fontweight','bold','fontsize',18);
%% Browse large RP err
LARGE_RPERR_THRESH = 3;
[i1,iphyspt1] = find(rperr1>LARGE_RPERR_THRESH);
[i2,iphyspt2] = find(rperr1>LARGE_RPERR_THRESH);
i = [i1;i2];
iphyspt = [iphyspt1;iphyspt2];
[t(i,{'mov' 'frm'}) table(iphyspt)]

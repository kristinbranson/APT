function yRoiLo = shcropy(yCents,bbymax,sd,yRoiSz)
% yCents: [nx1] y-centroids
% bb: [nx4] bboxes
% sd: SD to use, will be approximate spread of centroids after crop
% yRoiSz: total y-size of cropped roi

assert(iscolumn(yCents) && iscolumn(bbymax) && numel(yCents)==numel(bbymax));

%sd = 19.5; % from view1, x-dir
ymid = (1+yRoiSz)/2; %ymid = 140.5; % centroid here would be dead-center
ymidceil = ceil(ymid);
%bbymax = bb(:,4);

tfFarLeft = yCents<=ymidceil; % includes ycv1 = 1..141. These centroids stay put (crop maximally to left)
tfFarRight = bbymax-yCents+1<=ymidceil; % 
tfMid = ~tfFarLeft & ~tfFarRight; 
ycvmid = yCents(tfMid);
bbymaxmid = bbymax(tfMid);
dyLeft = ycvmid-ymidceil; 
assert(all(dyLeft>0));
dyRight = bbymaxmid-ymidceil+1-ycvmid;
assert(all(dyRight)>0);
maxJitterMid = min(dyLeft,dyRight);
actualJitterMid = arrayfun(@(x)randntruncated(sd,x),maxJitterMid);

fprintf(1,'%d far left, %d far right, %d mid\n',nnz(tfFarLeft),nnz(tfFarRight),nnz(tfMid));

% just check we didn't jitter into the endzones
ycvmidJittered = ycvmid + actualJitterMid;
assert(~any(ycvmidJittered<=ymidceil));
assert(~any(bbymaxmid-ycvmidJittered+1<=ymidceil));
fprintf(1,'Minimum ymidJittered: %.3f\n',min(ycvmidJittered));
fprintf(1,'Minimum bbymaxmid-ymidJittered+1: %.3f\n',...
  min(bbymaxmid-ycvmidJittered+1));

ycvmidJittered = ycvmid - actualJitterMid;
assert(~any(ycvmidJittered<=ymidceil));
assert(~any(bbymaxmid-ycvmidJittered+1<=ymidceil));
fprintf(1,'Minimum ymidJittered: %.3f\n',min(ycvmidJittered));
fprintf(1,'Minimum bbymaxmid-ymidJittered+1: %.3f\n',...
  min(bbymaxmid-ycvmidJittered+1));

yRoiLo = ones(size(yCents));
% 1 is correct for tfFarLeft
yRoiLo(tfFarRight) = bbymax(tfFarRight)-yRoiSz+1;
yRoiLo(tfMid) = ycvmid-ymid+1-actualJitterMid; % +/-actualJitterMid, shouldn't matter
yRoiLo = round(yRoiLo);
assert(all(yRoiLo>=1));
assert(all(bbymax-yRoiLo+1>=yRoiSz));

ycvroi = yCents - yRoiLo;
figure;
hist(ycvroi);
tstr = sprintf('ycvroi. sd=%.3f',std(ycvroi));
title(tstr,'fontweight','bold');
xlim([0 280]);
xlabel('y','fontweight','bold');

function xRoiLo = shcropx(xCents,xRoiSz,xFarLeftThresh)

xmid = xRoiSz/2; % a little sloppier here than in shcropy but shouldn't matter

fprintf('Median of non-far-left: %.3f\n',median(xCents(xCents>xFarLeftThresh)));

tfFarLeft = xCents<xFarLeftThresh;
tfRegLeft = xFarLeftThresh<=xCents & xCents<=xmid; % normal distro, left half of shapes (close to x=0)
xRegLeft = xCents(tfRegLeft);
tfFarRight = xCents>xmid+(xmid-xFarLeftThresh); % too far right. redistribute these
xFarRight = xCents(tfFarRight);
nFarRight = numel(xFarRight);
xFarRightDesired = xmid+xmid-randsample(xRegLeft,nFarRight);
fprintf('%d far lefts, %d reg lefts, redistrib %d far rights\n',...
  nnz(tfFarLeft),nnz(tfRegLeft),nFarRight);

xCents_new = xCents;
xCents_new(tfFarRight) = xFarRightDesired;
dxc = xCents_new-xCents;
assert(all(dxc<=0)); % only moving right-half shapes leftwards
xRoiLo = ones(size(xCents));
xRoiLo = xRoiLo-round(dxc);
assert(all(xRoiLo>=1));

figure;
hist(xCents_new);
tstr = sprintf('SD of xCents_new=%.3f\n',std(xCents_new));
title(tstr,'fontweight','bold');
xlabel('x','fontweight','bold');
xlim([0 xRoiSz]);
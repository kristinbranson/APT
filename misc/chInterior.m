function tfint = chInterior(xyCH,xgv,ygv)
[xx,yy] = meshgrid(xgv,ygv);
tfint = arrayfun(@(z1,z2)chIsInt(xyCH,z1,z2,1),xx,yy);
  
function P2 = readpdf2chan(P0,xg0,yg0,xg1,yg1,xc,yc,th)
nchan = size(P0,3);
for ichan=nchan:-1:1
  P2(:,:,ichan) = readpdf2(P0(:,:,ichan),xg0,yg0,xg1,yg1,xc,yc,th);
end
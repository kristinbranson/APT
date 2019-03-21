function [sbest,Xbest,xy1best,xy2best,hm1roi,hm2roi] = ...
  viewpref3dreconhmap(loghm1,loghm2,cr,varargin)
% hm*: NORMALIZED heatmaps

[dxyz,lambda2,hm1roi,hm2roi] = myparse(varargin,...
  'dxyz',0.01,...
  'lambda2',1,...
  'hm1roi',[],...
  'hm2roi',[]...
  );

tfROI = ~isempty(hm1roi);
assert(tfROI==~isempty(hm2roi));
if ~tfROI
  assert(false,'unsupported atm');
  fprintf('Finding roi from nz entries of hmaps.\n');
  hm1roi = findhmroi(loghm1);
  hm2roi = findhmroi(loghm2);
end
corners1 = getcorners(hm1roi);
corners2 = getcorners(hm2roi);

Xcorners = nan(3,16);
assert(isequal(4,size(corners1,1),size(corners2,1)));
for i1=1:4
for i2=1:4
  i = (i1-1)*4 + i2;
  Xcorners(:,i) = cr.stereoTriangulate(corners1(i1,:)',corners2(i2,:)');
end
end

xyzmin = min(Xcorners,[],2);
xyzmax = max(Xcorners,[],2);
xgv = xyzmin(1):dxyz:xyzmax(1);
ygv = xyzmin(2):dxyz:xyzmax(2);
zgv = xyzmin(3):dxyz:xyzmax(3);
fprintf(1,'xyz min max: %s %s\n',mat2str(xyzmin(:)'),mat2str(xyzmax(:)'));
fprintf(1,'3d grid size is %d %d %d\n',numel(xgv),numel(ygv),numel(zgv));

[x,y,z] = meshgrid(xgv,ygv,zgv);
[xy1_o,xy2_o] = proj12(cr,[xgv(1) ygv(1) zgv(1)]);
[xy1_x,xy2_x] = proj12(cr,[xgv(2) ygv(1) zgv(1)]);
[xy1_y,xy2_y] = proj12(cr,[xgv(1) ygv(2) zgv(1)]);
[xy1_z,xy2_z] = proj12(cr,[xgv(1) ygv(1) zgv(2)]);
dox_1 = sqrt(sum((xy1_o-xy1_x).^2));
doy_1 = sqrt(sum((xy1_o-xy1_y).^2));
doz_1 = sqrt(sum((xy1_o-xy1_z).^2));
dox_2 = sqrt(sum((xy2_o-xy2_x).^2));
doy_2 = sqrt(sum((xy2_o-xy2_y).^2));
doz_2 = sqrt(sum((xy2_o-xy2_z).^2));
fprintf(1,'dox1/y/z: %.2f %.2f %.2f. dox2/y/z: %.2f %.2f %.2f\n',...
  dox_1,doy_1,doz_1,dox_2,doy_2,doz_2);  

Xg1 = [x(:) y(:) z(:)]; % [ngx3]. Xgrid, cam1 coord sys
ng = size(Xg1,1);

[xy1,xy2] = proj12(cr,Xg1);

szassert(xy1,[ng 2]);
szassert(xy2,[ng 2]);
[hmnr1,hmnc1] = size(loghm1);
[hmnr2,hmnc2] = size(loghm2);

tfIB1 = checkIB(xy1,hmnr1,hmnc1);
tfIB2 = checkIB(xy2,hmnr2,hmnc2);
tfIB = tfIB1 & tfIB2;
fprintf(1,'%d candidates, %d IB1, %d IB2, %d IB\n',ng,...
  nnz(tfIB1),nnz(tfIB2),nnz(tfIB));

nib = nnz(tfIB);

Xg1ib = Xg1(tfIB,:);
xy1ib = xy1(tfIB,:);
xy2ib = xy2(tfIB,:);
% intentionally choose nearest here, for case where we are conf in 1 and 
% not in the other; and where RP err is high. In this case, the solns will
% lie in the "plateau" of view2 (lo conf) and we want the radius falloff
% cost to matter. if we choose bilinear, then details of 3d discretization
% will lead to better performance just by random chance in the hi-conf
% view.
loghm1ib = interp2(loghm1,xy1ib(:,1),xy1ib(:,2),'nearest'); 
loghm2ib = interp2(loghm2,xy2ib(:,1),xy2ib(:,2),'nearest');
szassert(loghm1ib,[nib 1]);
szassert(loghm2ib,[nib 1]);

score = loghm1ib + lambda2*loghm2ib;
[sbest,idx] = max(score);
Xbest = Xg1ib(idx,:);
xy1best = xy1ib(idx,:);
xy2best = xy2ib(idx,:);

function [xy1,xy2] = proj12(cr,Xg)
% Xg: [Nx3]
% xy1/2: [Nx2]

sp = cr.stroParams;
R = sp.RotationOfCamera2;
T = sp.TranslationOfCamera2;
xy2 = worldToImage(sp.CameraParameters2,R,T,Xg,'applyDistortion',true);
R = eye(3);
T = [0 0 0];
xy1 = worldToImage(sp.CameraParameters1,R,T,Xg,'applyDistortion',true);

function tfIB = checkIB(xy2d,nr,nc)
roi = [1 nc 1 nr];
tfIB = roi(1)<=xy2d(:,1) & xy2d(:,1)<=roi(2) & ...
       roi(3)<=xy2d(:,2) & xy2d(:,2)<=roi(4);

function xy = getcorners(roi)
xlo = roi(1);
xhi = roi(2);
ylo = roi(3);
yhi = roi(4);
xy = [ xlo ylo; xlo yhi; xhi ylo; xhi yhi ];
  
function roi = findhmroi(hm)
any1 = any(hm,1);
any2 = any(hm,2);
xlo = find(any1,1,'first');
xhi = find(any1,1,'last');
ylo = find(any2,1,'first');
yhi = find(any2,1,'last');
roi = [xlo xhi ylo yhi];



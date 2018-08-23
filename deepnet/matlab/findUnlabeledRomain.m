Q = load('/groups/branson/bransonlab/mayank/PoseTF/RomainLeg/Jun22Sep16Sep15Sep13Aug26Sep07Sep05.lbl','-mat');

%%

jj = [];
for ndx = 1:numel(Q.labeledpos)
  valid = squeeze(~all(isnan(Q.labeledpos{ndx}(:,1,:)),1));
  jj = cat(3,jj, Q.labeledpos{ndx}(:,:,valid));
end

%%

p1 = jj(:,1,:); p1 = p1(:);
p2 = jj(:,2,:); p2 = p2(:);

%%
figure; scatter(p1,p2,'.');

%%

msz = cell(numel(Q.labeledpos),3);
parfor ndx = 1:numel(Q.labeledpos)
  for view = 1:3
    [a,b,c,d] = get_readframe_fcn(Q.movieFilesAll{ndx,view});
    msz{ndx,view} = [d.nc,d.nr];
  end
end

%%

proj_err = {};
valid_frames = {};
for ndx = 1:numel(Q.labeledpos)
  
  %%
  curpts = Q.labeledpos{ndx}(:,:,:);
  cRig = Q.viewCalibrationData(ndx);
  
  valid = find(~all(isnan(curpts(:,1,:)),1));
  valid_frames{ndx} = valid;
  nfs = numel(valid);
  vpts = curpts(:,:,valid);
  
  vpts = permute(vpts,[3,1,2]);
  vptsL = reshape(vpts(:,1:18,:),[nfs*18 2]);
  
  vptsR = reshape(vpts(:,(1:18)+18,:),[nfs*18 2]);
  vptsB = reshape(vpts(:,(1:18)+36,:),[nfs*18 2]);

  dd = [];
  for idx = 1:size(vptsR,1)
    cropL = cRig.y2x([vptsL(idx,2) vptsL(idx,1)],'L');
    cropB = cRig.y2x([vptsB(idx,2) vptsB(idx,1)],'B');
    [P3d_L,P3d_B] = cRig.stereoTriangulate(cropL,cropB,'L','B');
    [rL_re,cL_re] = cRig.projectCPR(P3d_L,1);
    [rB_re,cB_re] = cRig.projectCPR(P3d_B,3);
    [rR_re,cR_re] = cRig.projectCPR(cRig.camxform(P3d_L,'LR'),2);
    dd(idx) = sqrt(sum((vptsR(idx,:) - [cR_re rR_re]).^2));
  end    
  proj_err{ndx} = reshape(dd,[nfs,18]);
  
end
  
%%

tr = 100;

ndx = 1;

zz = proj_err{ndx};
kk = find(zz(:)>70);
[aa,bb] = ind2sub(size(zz),kk);
curpts = Q.labeledpos{ndx}(:,:,valid_frames{ndx});
szL = msz{ndx,1};
szR = msz{ndx,2};
szB = msz{ndx,3};
corner = zeros(numel(aa),1);
mind = zeros(numel(aa),1);
for idy = 1:numel(aa)
  d2pt = zeros(1,3);
  for view = 1:3
    pt = curpts(bb(idy)+(view-1)*18,:,aa(idy));
    d2x = min(pt(1),msz{ndx,view}(1)-pt(1));
    d2y = min(pt(2),msz{ndx,view}(2)-pt(2));
    d2pt(view) = sqrt(d2x^2 + d2y^2);
  end
  corner(idy) = argmin(d2pt);
  mind(idy) = min(d2pt);
  
end

%%

% rr = randsample(numel(aa),30);
rr = find(mind>130);
for idy = rr(:)'
  fprintf('frame:%d pt:%d, view:%d\n',valid(aa(idy)),bb(idy),corner(idy));
  
end

%%

ndx = 1;
rfn = {};
for view = 1:3
  [a,b,c,d] = get_readframe_fcn(Q.movieFilesAll{ndx,view});
  rfn{view} = a;
end

%%

ndx = 1;
figure; 
curpts = Q.labeledpos{ndx}(:,:,:);
valid = find(~all(isnan(curpts(:,1,:)),1));

for view = 1:3
  subplot(1,3,view);
  im = rfn{view}(valid(1));
  imshow(im);
  hold on;
  scatter(curpts( (1:18)+(view-1)*18,1,valid(1)),curpts( (1:18)+(view-1)*18,2,valid(1)),'.');
  hold off;
  axis on;
end
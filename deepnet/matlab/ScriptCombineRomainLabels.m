% l1 = load('/home/mayank/work/poseEstimation/RomainLeg/Apr28AndJun22_onlyBottom.lbl','-mat');
l1 = load('/localhome/kabram/Dropbox (HHMI)/MultiViewFlyLegTracking/multiview labeling/romainJun22NewLabels.lbl','-mat');

lfiles1 = {'/localhome/kabram/Dropbox (HHMI)/MultiViewFlyLegTracking/sep1616/sep1616-1531Romain.lbl'
  '/localhome/kabram//Dropbox (HHMI)/MultiViewFlyLegTracking/sep1516/sep1516-1537Romain.lbl'
  '/localhome/kabram//Dropbox (HHMI)/MultiViewFlyLegTracking/sep1316/sep1316-1606Romain.lbl'};
lfiles2={'/localhome/kabram//Dropbox (HHMI)/MultiViewFlyLegTracking/ToLabel/sep0716/sep0716-1431_mayankLabeled.lbl'
  '/localhome/kabram//Dropbox (HHMI)/MultiViewFlyLegTracking/ToLabel/aug2616/aug2616-1120_mayankLabeled.lbl'
  '/localhome/kabram//Dropbox (HHMI)/MultiViewFlyLegTracking/ToLabel/sep0516/sep0516-1023_mayankLabeled.lbl'
  };

l = l1;
for ndx = 1:3
  l.movieFilesAll{ndx} = ['/localhome/kabram' l1.movieFilesAll{ndx}(18:end)];
  l.movieInfoAll{ndx} = l1.movieInfoAll{ndx};
  
end
l.labeledpos{1}(19:19:end,:,:) = [];
l.labeledpostag{1}(19:19:end,:) = [];
l.labeledposTS{1}(19:19:end,:) = [];
l.labeledposMarked{1}(19:19:end,:) = [];
l.labeledpos2{1}(19:19:end,:,:) = [];

count = 2;
for ndx = 1:numel(lfiles1)
  l2 = load(lfiles1{ndx},'-mat');
  for idx = 1:3
    l.movieFilesAll{count,idx} = ['/localhome/kabram' l2.movieFilesAll{idx}(18:end)];
  end
  l.labeledpos{count,1} = l2.labeledpos{1};
  l.labeledpostag{count,1} =l2.labeledpostag{1};
  l.labeledposTS{count,1} = l2.labeldposTS{1};
  l.labeledposMarked{count,1} = l2.labeledposMarked{1};
  l.labeledpos2{count,1} = l2.labeledpos2{1};
  l.viewCalibrationData(count) = l2.viewCalibrationData(1);
  count = count+1;
end

for ndx = 1:numel(lfiles2)
  l2 = load(lfiles2{ndx},'-mat');
  for idx = 1:3
    l.movieFilesAll{count,idx} = ['/localhome/kabram/Dropbox (HHMI)/MultiViewFlyLegTracking' strrep(l2.movieFilesAll{idx}(33:end),'\','/')];
  end
  l.labeledpos{count,1} = l2.labeledpos{1};
  if iscell(l2.viewCalibrationData)
    l.viewCalibrationData(count) = l2.viewCalibrationData{1};
  else
    l.viewCalibrationData(count) = l2.viewCalibrationData(1);
  end
  count = count+1;
end

%
% for ndx = 1:numel(l.movieFilesAll)
%   l.movieFilesAll{ndx} = ['/localhome/kabram/' l.movieFilesAll{ndx}(21:end)];
% end
l.cfg.NumLabelPoints = 18;

%% make nan pts that are in the corner.

Q = l;
msz = cell(numel(Q.labeledpos),3);
parfor ndx = 1:numel(Q.labeledpos)
  for view = 1:3
    [a,b,c,d] = get_readframe_fcn(Q.movieFilesAll{ndx,view});
    msz{ndx,view} = [d.nc,d.nr];
  end
end

%

proj_err = {};
valid_frames = {};
for ndx = 1:numel(Q.labeledpos)
  
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


for ndx = 1:numel(Q.labeledpos)

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
    cur_corner =  argmin(d2pt);
    corner(idy) = argmin(d2pt);
    mind(idy) = min(d2pt);
    
    Q.labeledpos{ndx}(bb(idy)+(cur_corner-1)*18,:,valid_frames{ndx}(aa(idy))) = nan;

  end

end


%%
save('/groups/branson/bransonlab/mayank/PoseTF/RomainLeg/Jun22Sep16Sep15Sep13Aug26Sep07Sep05_fixed.lbl','-struct','Q','-v7.3');


%%

lfiles1 = {'/localhome/kabram/Dropbox (HHMI)/MultiViewFlyLegTracking/sep1616/sep1616-1531Romain.lbl'
  '/localhome/kabram//Dropbox (HHMI)/MultiViewFlyLegTracking/sep1516/sep1516-1537Romain.lbl'
  '/localhome/kabram//Dropbox (HHMI)/MultiViewFlyLegTracking/sep1316/sep1316-1606Romain.lbl'
  '/localhome/kabram//Dropbox (HHMI)/MultiViewFlyLegTracking/ToLabel/sep0716/sep0716-1431_mayank.lbl'
  '/localhome/kabram//Dropbox (HHMI)/MultiViewFlyLegTracking/ToLabel/aug2616/aug2616-1120_labeled.lbl'
  };

for ndx = 1:numel(lfiles1)
  l2 = load(lfiles1{ndx},'-mat');
  fprintf('%s\n',l2.movieFilesAll{1})
end


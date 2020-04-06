l1 = load('/home/mayank/work/poseEstimation/RomainLeg/Apr28AndJun22_onlyBottom.lbl','-mat');
lfiles1 = {'/home/mayank/Dropbox/MultiViewFlyLegTracking/sep1616/sep1616-1531Romain.lbl'
  '/home/mayank/Dropbox/MultiViewFlyLegTracking/sep1516/sep1516-1537Romain.lbl'
  '/home/mayank/Dropbox/MultiViewFlyLegTracking/sep1316/sep1316-1606Romain.lbl'
  };
lfiles2 = {
  '/home/mayank/Dropbox (HHMI)/MultiViewFlyLegTracking/ToLabel/sep0716/sep0716-1431_mayank.lbl'
  '/home/mayank/Dropbox (HHMI)/MultiViewFlyLegTracking/ToLabel/aug2616/aug2616-1120_labeled.lbl'
  };

l = l1;
l.movieFilesAll{2} = '/home/mayank/Dropbox/MultiViewFlyLegTracking/older stuff/trackingApril28-14-53/bias_video_cam_2_date_2016_04_28_time_14_53_16_v001.avi';
l.movieFilesAll{3} = '/home/mayank/Dropbox/MultiViewFlyLegTracking/older stuff/trackingApril28-15-23/bias_video_cam_2_date_2016_04_28_time_15_23_20_v001.avi';

for ndx = 1:numel(lfiles1),
  l2 = load(lfiles1{ndx},'-mat');

  l.movieFilesAll{end+1} = ['/home/mayank/Dropbox/' strrep(l2.movieFilesAll{3}(33:end),'\','/')];
  
  l.labeledpos{end+1} = l2.labeledpos{1}(37:end,:,:);
end

for ndx = 1:numel(lfiles2),
  l2 = load(lfiles2{ndx},'-mat');

  l.movieFilesAll{end+1} = ['/home/mayank/Dropbox/MultiViewFlyLegTracking' strrep(l2.movieFilesAll{3}(33:end),'\','/')];
  
  l.labeledpos{end+1} = l2.labeledpos{1}(37:end,:,:);
end

for ndx = 1:numel(l.movieFilesAll)
  l.movieFilesAll{ndx} = ['/home/mayank/Dropbox (HHMI)' l.movieFilesAll{ndx}(21:end)];
end

save('/home/mayank/work/poseEstimation/RomainLeg/Apr28Jun22Sep16Sep15Sep13Aug26Sep07_onlyBottom.lbl','-struct','l','-v7.3');


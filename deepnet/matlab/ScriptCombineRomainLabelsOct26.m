Qmv = load('/localhome/kabram/Dropbox (HHMI)/MultiViewFlyLegTracking/multiview labeling/romainJun22NewLabels.lbl','-mat');
Qs1 = load('/localhome/kabram/Dropbox (HHMI)/MultiViewFlyLegTracking/older stuff/trackingApril28-14-53/track_date_2016_04_28_time_14_53_16.lbl','-mat');
Qs2 = load('/localhome/kabram/Dropbox (HHMI)/MultiViewFlyLegTracking/older stuff/trackingApril28-15-23/track_date_2016_04_28_time_15_23.lbl','-mat');

%%

Q = struct;
Q.movieFilesAll{1,1} = Qmv.movieFilesAll{3};
Q.movieFilesAll{2,1} = Qs1.movieFilesAll{1};
Q.movieFilesAll{3,1} = Qs2.movieFilesAll{3};

Q.labeledpos{1,1} = Qmv.labeledpos{1}(end-18:end-1,:,:);
Q.labeledpos{2,1} = Qs1.labeledpos{1}(1:18,:,:);
Q.labeledpos{3,1} = Qs2.labeledpos{3}(1:18,:,:);

Q.cfg.NumLabelPoints = 19;

for ndx = 1:3
  Q.movieFilesAll{ndx,1} = fullfile('/localhome/kabram/Dropbox (HHMI)/',Q.movieFilesAll{ndx,1}(33:end));
  xx = Q.labeledpos{ndx}(:,1,:);
  infx = squeeze(any(isinf(xx),1));
  Q.labeledpos{ndx}(:,:,infx) = nan;
end

save('/groups/branson/bransonlab/mayank/PoseTF/RomainLeg/Apr28AndJun22.lbl','-struct','Q','-v7.3');

%%

for ndx = 1:3
  xx = Q.labeledpos{ndx}(:,1,:);
  nanx = isnan(squeeze(xx(1,1,:)));
  infx = squeeze(any(isinf(xx),1));
  xx = xx(:,1,~(nanx|infx));
  minx = min(xx(:));
  maxx = max(xx(:));
  yy = Q.labeledpos{ndx}(:,2,:);
  nany = isnan(squeeze(yy(1,:)));
  infy = squeeze(any(isinf(yy),1));
  yy = yy(:,~(nanx|infy));
  miny = min(yy(:));
  maxy = max(yy(:));
  [a,b,c,d] = get_readframe_fcn(Q.movieFilesAll{ndx});
  
  fprintf('minx:%.2f maxx:%.2f miny:%.2f maxy:%.2f width:%d,height:%d\n',...
    minx,maxx,miny,maxy,d.nc,d.nr);
end

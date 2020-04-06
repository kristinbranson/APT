Q = load('~/bransonlab/PoseTF/janLegTracking/35exps@20160421.lbl','-mat');
expdirs = {};
pts = [];
expidx = [];
ts = [];
for ndx = 1:numel(Q.movieFilesAll)
  expdirs{ndx} = ['/groups/branson/bransonlab/mayank/PoseEstimationData/JanLegTracking/' Q.movieFilesAll{ndx}(13:end)];
  lts = find(~any(any(isnan(Q.labeledpos{ndx}),1),2));
  ts = [ts lts'];
  expidx = [expidx repmat(ndx,1,numel(lts))];
  pts = cat(3,pts,Q.labeledpos{ndx}(4:7,:,lts));
end
J = struct;
J.expdirs = expdirs;
J.expidx = expidx;
J.pts = permute(pts,[2 4 1 3]);
J.ts = ts;
J.vid1files = expdirs;

save('janLegTracking/Labels20160421.mat','-struct','J','-v7.3');
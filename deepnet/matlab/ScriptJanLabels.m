Q = load('/groups/branson/bransonlab/mayank/PoseTF/janLegTracking/160819_Dhyey_2_al_retrain20160911_v7_cleaned20161101.lbl','-mat');

%%

nmovies = numel(Q.labeledpos);
newm = cell(nmovies,1);
numLabels = [];
flyId = {};
missing = false(1,nmovies);

for ndx = 1:numel(Q.movieFilesAll)
  curmov = Q.movieFilesAll{ndx};
  if startsWith(curmov,'$dataroot')
    curmov = ['/groups/card/cardlab/Jan2Allen_Tracking/data' curmov(10:end)];
  elseif startsWith(curmov,'$root151106_03')
    curmov = ['/groups/card/cardlab/Jan2Allen_Tracking/151106_03' curmov(15:end)];
  else
    fprintf('%d %s doesnt start with anything\n',ndx,curmov);
  end
  
  curmov = strrep(curmov,'\','/');
  if ~exist(curmov,'file')
    fprintf('%d %s doesnt exist\n',ndx,curmov);
    missing(ndx)=true;
  end
  newm{ndx} = curmov;
  numLabels(ndx) = nnz(all(all(~isnan(Q.labeledpos{ndx}),1),2));
  ss = strsplit(curmov,'/');
  flyId{ndx} = ss{end}(1:9);
end
Q.movieFilesAll = newm;

%%

u_fly = unique(flyId);
labels_per_fly = [];
for ndx = 1:numel(u_fly)
  fly_ndx = strcmp(u_fly{ndx},flyId);
  labels_per_fly(ndx) = sum(numLabels(fly_ndx));
end
%%

dd = fieldnames(Q);
for ndx = 1:numel(dd)
  if size(Q.(dd{ndx}),1)==280
    fprintf('%s\n',dd{ndx});
    Q.(dd{ndx})(missing,:) = [];
  end
end

%%

save('/groups/branson/home/kabram/bransonlab/PoseTF/janLegTracking/160819_Dhyey_2_al_fixed.lbl','-struct','Q','-v7.3');

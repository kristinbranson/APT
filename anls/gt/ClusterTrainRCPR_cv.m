function rc = ClusterTrainRCPR_cv(matfile,cvi,v,resfile)

load(matfile,'IsAll','bboxesAll','cvd','ptsAll','expidxAll','sPrm');
try
  load(matfile,'ncores');
end
if ischar(cvi),
  cvi = str2double(cvi);
end
if ischar(v),
  v = str2double(v);
end

if exist('ncores','var'),
  maxNumCompThreads(ncores);
  res = maxNumCompThreads;
  fprintf('Set maximum number of threads to %d\n',res);
end

% exclude test data
NtrainAll = size(IsAll,1);
istrain = true(1,NtrainAll);
istrain(ismember(expidxAll,cvd.split{cvi})) = false;
Ntrn = nnz(istrain);

%sPrm = ReadYaml(paramfile);

fprintf('Split %d, view %d, ntrain = %d\n',cvi,v,Ntrn);
% initialize regressor
rc = RegressorCascade(sPrm);
rc.init();

rc.trainWithRandInit(IsAll(istrain,v),bboxesAll(istrain,:,v),ptsAll(istrain,:,v));
%pAll = reshape(pAll,Ntrn,sPrm.TrainInit.Naug,nlandmarks*2,rc.nMajor+1);
save(resfile,'rc','cvi','v','matfile');

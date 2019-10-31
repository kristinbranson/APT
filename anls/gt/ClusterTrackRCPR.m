function ClusterTrackRCPR(moviefile,cvi,v,f0,f1,matfile,resfile,ncores)

if ischar(cvi),
  cvi = str2double(cvi);
end
if ischar(v)
  v = str2double(v);
end
if ischar(f0),
  f0 = str2double(f0);
end
if ischar(f1),
  f1 = str2double(f1);
end

if exist('ncores','var'),
  if ischar(ncores),
    ncores = str2double(ncores);
  end
  maxNumCompThreads(ncores);
  res = maxNumCompThreads;
  fprintf('Set maximum number of threads to %d\n',res);
end

load(matfile,'sPrm','allrcs');

[readframe,nframes] = get_readframe_fcn(moviefile);
f1 = min(nframes,f1);
nframes = f1-f0+1;

IsTest = cell(nframes,1);
% R{v}.cpr_2d_locs is nframes x nlandmarks x 2
%R{v}.cpr_2d_locs = nan([nframes,nlandmarks,2]);

[nr,nc,ncolors] = size(readframe(1));
    
for i = 1:nframes,
  f = f0+i-1;
  IsTest{i} = readframe(f);
  if ncolors > 1,
    IsTest{i} = rgb2gray(IsTest{i});
  end
end
   
nlandmarks = allrcs{cvi,v}.prmModel.nfids; %#ok<USENS>
pTmp = allrcs{cvi,v}.propagateRandInit(IsTest,repmat([1,1,nc,nr],[nframes,1]),sPrm.TestInit);
pTmp = reshape(pTmp,nframes,sPrm.TestInit.Nrep,nlandmarks*2,allrcs{cvi,v}.nMajor+1);
% nframes x nreps x nviews*nlandmarks x (niters+1)
pTmp = permute(pTmp(:,:,:,end),[1,3,2]);
% nframes x 2*nlandmarks
pSelectedTmp = rcprTestSelectOutput(pTmp,allrcs{cvi,v}.prmModel,sPrm.Prune);
% nframes x nlandmarks x 2
pSelectedTmp = reshape(pSelectedTmp,[nframes,nlandmarks,2]);

R = struct;
R.cpr_2d_locs = pSelectedTmp;
R.cpr_all2d_locs = reshape(pTmp,[nframes,nlandmarks,2,sPrm.TestInit.Nrep]);
R.movie = moviefile;

save(resfile,'-struct','R');
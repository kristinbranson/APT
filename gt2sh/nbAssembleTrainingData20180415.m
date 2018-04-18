%%
load tblFinalReconciled_20180415T212437.mat;

%% tFinalReconciled check movpaths/massage
mfa = tFinalReconciled.movFile_m;
e = cellfun(@(x)exist(x,'file'),mfa);
nnz(e==0)
mfa(e==0)
tf = e==0;
%% tFinalReconciled check movpaths/massage
pat = '/groups/huston/hustonlab/flp-chrimson_experiments/fly_229_to_238_1st_to_2nd_12_15_norpAchrimsonFLP_SS002323/fly230/fly230_300msStimuli';
replace = '/groups/huston/hustonlab/flp-chrimson_experiments/fly_229_to_238_1st_to_2nd_12_15_norpAchrimsonFLP_SS002323/fly230_300msStimuli';
mfabad = mfa(tf)
assert(all(strncmp(mfabad,pat,numel(pat))))
mfa(tf) = regexprep(mfa(tf),pat,replace);
mfa(tf)
%% tFinalReconciled check movpaths/massage
tFinalReconciled.movFile_read = mfa;
e = cellfun(@(x)exist(x,'file'),mfa);
nnz(e==0)

%% tGT check movpaths/massage
mfa = tGT.movFile;
flpIdx = cellfun(@(x)regexp(x,'flp-chrimson_experiments'),mfa,'uni',0);
nflpIdx = cellfun(@numel,flpIdx);
nnz(nflpIdx~=1)

flpIdx = cell2mat(flpIdx);
figure
hist(flpIdx(:))
tf = flpIdx>5;
mfa(tf)

for i=1:numel(mfa)
  mfa{i} = ['/groups/huston/hustonlab/' mfa{i}(flpIdx(i):end)];
end
e = cellfun(@(x)exist(x,'file'),mfa);
nnz(e==0)
mfa(e==0)
%% tGT check movpaths/massage 2
ROOT = '/groups/huston/hustonlab/flp-chrimson_experiments';
TOFROM = {...
  'fly_450_to_452_26_9_16' 'fly_450_to_452_26_9_16norpAkirPONMNchr'
  'fly_453_to_457_27_9_16' 'fly_453_to_457_27_9_16_norpAkirPONMNchr'
  'fly_920_to_924_flpchr12D05lexA_SS47483kir' 'fly_920_to_924_flpchr12D05lexA_SS47483kir_badStockDidntWork'
  'fly_925_to_929_30_1_18_12D05flpChr_SS47492kir' 'fly_925_to_929_30_1_18_12D05flpChr_SS47492kir_badStockDidntWork'
  'fly_930_to_934_31_1_18_35G07flpChr_SS47483Kir' 'fly_930_to_934_31_1_18_35G07flpChr_SS47483Kir_badStockDidntWork'
  'fly_935_to_939_5_2_18_12D05flpChr_SS47483kir' 'fly_935_to_939_5_2_18_12D05flpChr_SS47483kir_badStockDidntWork'
  'fly_940_to_946_12D05lexAflpChr_SS47483Kir_6_2_18' 'fly_940_to_946_12D05lexAflpChr_SS47483Kir_6_2_18_badStockDidntWork'
  'fly_947_to_953_12D05lexAflpChr_SS47483kir_7_2_18' 'fly_947_to_953_12D05lexAflpChr_SS47483kir_7_2_18_badStockDidntWork'
};
NPAIRS = size(TOFROM,1);
for i=1:NPAIRS
  old = fullfile(ROOT,TOFROM{i,1});
  new = fullfile(ROOT,TOFROM{i,2});
  pat = ['^' old];  
  mfa2 = regexprep(mfa,pat,new);
  ndiff = nnz(~strcmp(mfa2,mfa));
  fprintf(1,'Pair %d, replaced %d movpaths.\n',i,ndiff);
  mfa = mfa2;
end

e = cellfun(@(x)exist(x,'file'),mfa);
nnz(e==0)
mfa(e==0)
tGT.movFile_read = mfa;

%% Save tables with new field movFile_read
nowstr = datestr(now,'yyyymmddTHHMMSS');
fname = sprintf('trnDataSH_%s.mat',nowstr);
save(fname,'tFinalReconciled','tGT');

%%
tRead = tFinalReconciled;

dryfile = sprintf('dry_%s',nowstr);
diary(dryfile);

tic;

n = height(tRead);
I = cell(n,2);
movsUn = unique(tRead.movFile_read(:));
% mr = MovieReader;
for iMov=1:numel(movsUn)
  mov = movsUn{iMov};
  % mr.open(mov);
  
  tf = strcmp(tRead.movFile_read,mov);
  [rows,vws] = find(tf);
  nIm = numel(rows);  
  maxfrm = max(tRead.frm(rows));  
  fprintf(1,'mov %d (%s). %d images to read. maxfrm is %d.\n',iMov,mov,nIm,maxfrm);

  imstack = readAllFrames(mov,maxfrm);
  
  for iIm=1:nIm
    trow = tRead(rows(iIm),:);
    %im = mr.readframe(trow.frm);
    im = imstack{trow.frm};    
    I{rows(iIm),vws(iIm)} = im;
  end
end

% matfile = sprintf('tblFinalReconciled_I_%s.mat',nowstr);
% save(matfile,'I');

toc;
diary off

%%
IFinalReconciled = I;
save trnDataSH_20180417T094303_IFinalReconciled.mat -v7.3 IFinalReconciled
  
  
  

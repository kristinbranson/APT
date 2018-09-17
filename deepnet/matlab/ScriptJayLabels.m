%% load the labels file

Q = load('/groups/branson/home/bransonk/tracking/code/rcpr/data/MousePaw/M118_M119_M122_M127_M130_M173_M174_M147_M194_20160310.mat');

%% convert to new lbl format with the nrs file locations

nmov = numel(Q.expdirs);
newnames = cell(nmov,1);
lclnames = cell(nmov,1);
nlabels = zeros(nmov,1);
mousenum = zeros(nmov,1);

missing = false(nmov,1);
mstr = {'20150209L','20150130L','20150129L'};
rstr = {'20150209Ltraining','20150130LnoFR','20150129LnoFR'};

lcldir = '/nrs/branson/mayank/jayData';
for ndx = 1:nmov
  curmov = Q.expdirs{ndx};
  curmov = ['/groups/hantman/hantmanlab/from_tier2/' curmov(15:end)];
  
  for idx = 1:numel(mstr)
    curmov = strrep(curmov,mstr{idx},rstr{idx});
  end
  
  expdir = curmov;
  curmov = fullfile(curmov,Q.moviefilestr);
  
  if ~exist(curmov,'file')
    curmov = [expdir '.avi'];
  end
  
  newnames{ndx} = curmov;
  nlabels(ndx) = nnz(Q.expidx == ndx);
  
  if ~exist(curmov,'file')
    fprintf('%d %s does not exist\n',ndx,curmov);
    missing(ndx,1) = true;
    continue;
  end
  
  pparts = strsplit(curmov,'/');
  
  idstr = '';
  if numel(pparts)>11,
    for ix = 11:numel(pparts)-1
      idstr = [idstr '/' pparts{ix}];
    end
  else
    for ix = 10:numel(pparts)-1
      idstr = [idstr '/' pparts{ix}];
    end
  end
  newloc = fullfile(lcldir,idstr);
  if ~exist(newloc,'dir')
    mkdir(newloc)
  end
  
  lclmovname = fullfile(newloc,pparts{end});
  if ~exist(lclmovname,'file')
    copyfile(newnames{ndx},lclmovname);
  end
  
  lclnames{ndx} = lclmovname;
  
end

%%

Q.expdirs = lclnames;

%%

R = struct;
R.movieFilesAll(:,1) = Q.expdirs;
R.movieFilesAll(:,2) = Q.expdirs;

nexp = numel(Q.expdirs);
all_nframes = zeros(nmov,1);
lpos = cell(nexp,1);
lmarked = cell(nexp,1);
lothers = cell(nexp,1);

for ndx = 1:nexp
  if isempty(Q.expdirs{ndx}), continue; end
    
  [rfn,~,fid,hinfo] = get_readframe_fcn(Q.expdirs{ndx});
  if hinfo.nc == 768 % remove movies with a different width.
    missing(ndx,1) = true;
  end
  
  all_nframes(ndx) = uint64(hinfo.nframes);
  
  nframes = int64(hinfo.nframes);
  curidx = find(Q.expidx==ndx);
  pside = permute(Q.pts(:,1,curidx),[2,1,3]);  
  pfront = permute(Q.pts(:,2,curidx),[2,1,3]);
  curpos = nan(2,2,nframes);
  curt = Q.ts(curidx);
  curpos(1,:,curt) = pside;
  curpos(2,:,curt) = pfront;
  lpos{ndx} = permute(curpos,[2,1,3]);
  curm = false(10,nframes);
  curm(:,curidx) = true;
  lmarked{ndx} = curm;
  if fid>0,fclose(fid); end
  
end
R.labeledpos = lpos; 
R.labeledposMarked = lmarked;
R.cfg.NumLabelPoints = 1;


%% Remove the missing files

dd = fieldnames(R);
nold = numel(newnames);
for ndx = 1:numel(dd)
  if size(R.(dd{ndx}),1)==nold
    fprintf('%s\n',dd{ndx});
    R.(dd{ndx})(missing,:) = [];
  end
end


%%

save('/groups/branson/bransonlab/mayank/PoseTF/data/jayMouse/miceLabels_20170412.lbl','-struct','R','-v7.3');

%%

Q = load('/groups/branson/bransonlab/mayank/PoseTF/data/jayMouse/miceLabels_20170412.lbl','-mat');
ht = [];
wd = [];
nlabels = [];
for ndx = 160:size(Q.movieFilesAll,1)
  [rfn,nframes,fid,hinfo] = get_readframe_fcn(Q.movieFilesAll{ndx,1});  
  ht(end+1) = hinfo.nr;
  wd(end+1) = hinfo.nc;
  if fid>0
    fclose(fid);
  end
  nlabels(end+1) = nnz(~all(isnan(Q.labeledpos{ndx}(:,1,:)),1));
end



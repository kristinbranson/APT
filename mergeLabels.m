mids = {'M118','M119','M122','M127','M130'};

nperexp = 3000;
origdatadir = '/home/mayank//Work/Tracking/data/mouse/Data';
origfiledir = '/home/mayank//Work/Tracking/data/mouse/explists';
multidatadir = '/home/mayank//Dropbox/AdamVideos/multiPoint';
outdir = '/home/mayank/Work/Tracking/data/mouse/Data';
for ndx = 1:numel(mids)
  % load all the data for orig point.
  fid = fopen(fullfile(origfiledir,['labeledexps_' mids{ndx} '.txt']));
  curl = fgetl(fid);
  O = {};
  Odirs = {};
  Ondx = [];
  Ondx1 = [];
  while ischar(curl)
    curl = strtrim(curl);
    [~,fname,~] = fileparts(curl);
    O{end+1} = load(fullfile(origdatadir,fname));
    curl = fgetl(fid);
    nfiles = numel(O{end}.expdirs);
    Odirs(end+1:end+nfiles) = O{end}.expdirs;
    Ondx(end+1:end+nfiles) = numel(O);
    Ondx1(end+1:end+nfiles) = 1:nfiles;
  end
  fclose(fid);
  
  % load all the data for multi point.
  dd1 = dir(fullfile(multidatadir,mids{ndx},[mids{ndx} 'a*.mat']));
  
  M = {};
  Mdirs = {};
  Mndx = [];Mndx1 = [];
  for idx = 1:numel(dd1)
    [~,fname,~] = fileparts(dd1(idx).name);
    wname = strrep(fname,'a','w');
    M{idx,1} = load(fullfile(multidatadir,mids{ndx},dd1(idx).name));
    M{idx,2} = load(fullfile(multidatadir,mids{ndx},[wname '.mat']));
    assert(isequal(M{idx,1}.expdirs,M{idx,2}.expdirs));
    nfiles = numel(M{idx,1}.expdirs);
    Mdirs(end+1:end+nfiles) = M{idx,1}.expdirs;
    Mndx(end+1:end+nfiles) = idx;
    Mndx1(end+1:end+nfiles) = 1:nfiles;
  end
  
  All = struct();
  count = 1;
  for idx1 = 1:numel(Mdirs)
    idx2 = find(strcmp(Mdirs{idx1},Odirs));
    if isempty(idx2), continue;end
    if isempty(O{Ondx(idx2)}.labeledpos_perexp{Ondx1(idx2)}), continue; end
    if isempty(M{Mndx(idx1),1}.labeledpos_perexp{Mndx1(idx1)}),continue; end
    if isempty(M{Mndx(idx1),2}.labeledpos_perexp{Mndx1(idx1)}),continue; end
    ll = cat(1,O{Ondx(idx2)}.labeledpos_perexp{Ondx1(idx2)},...
      M{Mndx(idx1),1}.labeledpos_perexp{Mndx1(idx1)},...
      M{Mndx(idx1),2}.labeledpos_perexp{Mndx1(idx1)});
    All.labeledpos_perexp{count} = ll;
    cure = Mdirs{idx1};
    cure = ['\tier2\hantman' cure(3:end)];
    cure = strrep(cure,'\','/');
    All.expdirs{count} = cure;
    count = count + 1;
  end
  save(fullfile(outdir,[mids{ndx} '_all.mat']),'-struct','All');
  
end
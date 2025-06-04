function SetLblFilePathRoot(oldlblfile,newlblfile,oldroot,newroot)

% oldlblfile = '/groups/branson/bransonlab/DataforAPT/JumpingMice/projects/202308avgc.lbl';
% newlblfile = '/groups/branson/home/bransonk/202308avgc.lbl';
% oldroot = 'X:\DataForAPT\JumpingMice'
% newroot = '/groups/branson/bransonlab/DataforAPT/JumpingMice'
% SetLblFilePathRoot(oldlblfile,newlblfile,oldroot,newroot)

obj = Labeler;
[success,tlbl,wasbundled] = obj.projUnbundleLoad(oldlblfile);
olds = load(tlbl,'-mat');

oldMacro = struct;
oldMacro.root = oldroot;
newMacro = struct;
newMacro.root = newroot;
news = olds;

fns = {'movieFilesAll','movieFilesAllGT','trxFilesAll','trxFilesAllGT'};

for fn1 = fns,
  fn = fns{1};
  if ~isfield(olds,fn),
    continue;
  end
  [~,pathstrMacroized] = FSPath.macrosPresent(olds.(fn),oldMacro);
  pathstrMacroized = pathstrMacroized{1};
  news.(fn)= FSPath.macroReplace(pathstrMacroized, newMacro);
end

save(tlbl,'-mat','-struct','news');
[rawLblFile,projtempdir] = obj.projGetRawLblFile();
allfiles = mydir(projtempdir);
tar([newlblfile '.tar'],allfiles);
movefile([newlblfile '.tar'],newlblfile);

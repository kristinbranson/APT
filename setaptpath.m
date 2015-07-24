function setaptpath

SUBDIRS = {
  'misc'
  'filehandling'
  'private_imuitools'
  };
      
root = aptroot;
p = [{root}; cellfun(@(x)fullfile(root,x),SUBDIRS,'uni',0)];
addpath(p{:});

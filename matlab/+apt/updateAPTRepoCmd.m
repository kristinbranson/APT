function [codestr, aptroot] = updateAPTRepoCmd(varargin)

[aptparent,downloadpretrained,branch] = ...
  myparse(varargin,...
          'aptparent','/home/ubuntu',...
          'downloadpretrained',false,...
          'branch','develop'... % branch to checkout
          );

aptroot = linux_fullfile(aptparent, 'APT') ;
deepnetroot = linux_fullfile(aptroot, 'deepnet') ;

codestr = {
  sprintf('cd %s ; ',deepnetroot) ;
  sprintf('git checkout %s ; ',branch) ;
  'git pull ; ' ;
  };
if downloadpretrained
  % assumes we are on lnx. we cd-ed into deepnet above
  codestr{end+1,1} = sprintf(DeepTracker.pretrained_download_script_py,deepnetroot);
end
codestr = cat(2,codestr{:});


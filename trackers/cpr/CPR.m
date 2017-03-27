classdef CPR
  
  properties (Constant)
    Root = fileparts(mfilename('fullpath'));
    MANIFESTFILE = 'Manifest.txt';
    Manifest = lclReadManifest( fullfile(CPR.Root,CPR.MANIFESTFILE) );
  end

  methods (Static)
  
    function [p,jp] = getpath()
      % p: cellstr, path
      % jp: cellstr, javapath
      
      m = CPR.Manifest;
      jctroot = m.jctrax;
      aptroot = m.apt;
      piotrroot = m.piotr;
      root = CPR.Root;
      
      addpath(aptroot);
      [aptpath,aptjpath] = APT.getpath;
      
      piotrpath = genpath(piotrroot);
      piotrpath = regexp(piotrpath,pathsep,'split');
      piotrpath = piotrpath(:);     
      tfRm = cellfun(@(x)~isempty(regexp(x,'__MACOSX','once')) || isempty(x),piotrpath);
      piotrpath(tfRm,:) = [];
      
      p = { ...
        root; ...
        fullfile(root,'misc'); ...
        fullfile(root,'video_tracking'); ...
        fullfile(root,'jan'); ...
        fullfile(root,'romain'); ...
        fullfile(jctroot,'misc'); ...
        fullfile(jctroot,'filehandling'); ...
        };
      p = [p;aptpath;piotrpath];
      
      jp = aptjpath;
    end
    
    function setpath
      [p,jp] = CPR.getpath();
      warnst = warning('off','MATLAB:dispatcher:pathWarning');
      addpath(p{:},'-begin');
      warning(warnst);      
      javaaddpath(jp);
    end
    
  end
  
end

function s = lclReadManifest(fname)
if exist(fname,'file')==0
  error('APT:Manifest','Cannot find Manifest file ''%s''. Please copy from Manifest.txt.sample and edit for your machine.',fname);
end
tmp = importdata(fname);
tmp = regexp(tmp,',','split');
tmp = cat(1,tmp{:});
s = cell2struct(tmp(:,2),tmp(:,1));
end
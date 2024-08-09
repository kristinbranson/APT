classdef Icons
  properties (Constant)
    GFXDIR = lclInitGfxDir();
    ims = lclInitIms(); % ims.keyword is [nxnx3] RGB icon image with nan for transparency
  end
end

function d = lclInitGfxDir()
d = fullfile(APT.getRoot,'gfx');
% if isdeployed,
% warndlg(sprintf('APTROOT %s',APT.getRoot))
% end
% PNGSHOULDEXIST = 'playsegmentblack.png';
% if exist(fullfile(d,PNGSHOULDEXIST),'file')==0
%   % might occur in compiled/deployed matlab. The gfx dir will be located in
%   % a different place in exploded ctf
%   d = which(PNGSHOULDEXIST); 
%   if isempty(d)
%     error('Could not find %s on path.',PNGSHOULDEXIST);
%   end
%   d = fileparts(d);
% end
end

function s = lclInitIms()
s = struct();
d = dir(fullfile(Icons.GFXDIR,'*.png'));
for i=1:numel(d)
  name = d(i).name;
  [~,namenoext] = fileparts(name);
  % AL: currently all gfx pngs ending in 'black' are read into Icons.m
  tfIsIconPng = isequal(namenoext(end-4:end),'black');
  if tfIsIconPng
    nameshort = namenoext(1:end-5);

    im = imread(fullfile(Icons.GFXDIR,name));
    im = double(im)/256;
    im(im==0) = nan;

    s.(nameshort) = im;
  end
end
%fprintf('Deployed: fieldnames ims: %s\n',String.cellstr2CommaSepList(fieldnames(s)));
end
    

function readscorefun = get_readscore_fcn(hmdir,fly,pti,varargin)

[hmtype,firstframe] = myparse(varargin,'hmtype','jpg','firstframe',1);

switch hmtype,

  case 'jpg',
  
    readscorefun = @(n) readscorefile(fullfile(hmdir,sprintf('hmap_trx_%d_t_%d_part_%d.jpg',fly,n+firstframe-1,pti)));
    
  case 'mjpg'
    [~,expname] = fileparts(hmdir);
    readscorefun = get_readframe_fcn(fullfile(hmdir,sprintf('%s_trx_%d_part_%d.avi',expname,fly,pti)));

  otherwise
    
    error('Not implemented');
    
end

function score = readscorefile(filename)

if ~exist(filename,'file'),
  error('File %s does not exist',filename);
end
score = im2double(imread(filename));
score = score - min(score(:));
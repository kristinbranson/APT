function scorefilenames = get_score_filenames(hmdir,fly,ptis,ns,varargin)

[hmtype,firstframe] = myparse(varargin,'hmtype','jpg','firstframe',1);

switch hmtype,
  
  case 'jpg',

    scorefilenames = cell([numel(ptis),numel(ns)]);
    
    for i = 1:numel(ptis),
      for j = 1:numel(ns),
        scorefilenames{i,j} = fullfile(hmdir,sprintf('hmap_trx_%d_t_%d_part_%d.jpg',fly,ns(j),ptis(i)));
      end
    end
    
  case 'mjpg',

    scorefilenames = cell([numel(ptis),1]);
    [~,expname] = fileparts(hmdir);
    
    for i = 1:numel(ptis),
      scorefilenames{i} = fullfile(hmdir,sprintf('%s_trx_%d_part_%d.avi',expname,fly,ptis(i)));
    end

  otherwise
    
    error('Not implemented');
    
end
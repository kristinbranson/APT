% [trx,matname,succeeded] = load_tracks(matname,[moviename])

function [trx,matname,succeeded,timestamps,rest] = load_tracks(matname,moviename,varargin)

[dosave,savename,annname,verbose] = myparse(varargin,'dosave',false,'savename','','annname','','verbose',false);

succeeded = false;
trx = [];
rest = struct;

isinteractive = ~exist('matname','var');

if isinteractive,
  helpmsg = 'Choose mat file containing trajectories to load';
  [matname,matpath] = uigetfilehelp('*.mat','Choose mat file containing trajectories','','helpmsg',helpmsg);
  if ~ischar(matname),
    return;
  end
  matname = [matpath,matname];
end

tmp = load(matname);
if verbose,
  fprintf('loaded %s\n',matname);
end
if isfield(tmp,'pairtrx'),
  tmp.trx = tmp.pairtrx;
end
if ~isfield(tmp,'trx'),
  if verbose,
    fprintf('no trx variable\n');
  end
  if isfield(tmp,'ntargets'),
    if verbose,
      fprintf('Ctrax output file; converting to trx file\n');
    end
    if ~exist('moviename','var'),
      moviename = '?';
    end
    %ds = datestr(now,30);
    if verbose,
      fprintf('Calling cleanup_ctrax_data\n');
    end
    [trx,matname,timestamps] = cleanup_ctrax_data(matname,moviename,tmp,'','dosave',dosave,'savename',savename,'annname',annname);
  else
    msgbox('Could not load data from %s, exiting',matname);
    return;
  end
else
  if verbose,
    fprintf('trx variable found\n');
  end
  trx = tmp.trx;
  
  nframes = max([trx.endframe]);
  timestamps = nan(1,nframes);
  if isfield(tmp,'timestamps'),
    timestamps = tmp.timestamps;
  elseif isfield(trx,'timestamps'),
    for i = 1:numel(trx),
      timestamps(trx(i).firstframe:trx(i).endframe) = trx(i).timestamps;
    end
  end
  
  fns = setdiff(fieldnames(tmp),{'trx','timestamps'});
  for i = 1:numel(fns),
    rest.(fns{i}) = tmp.(fns{i});
  end
  
  if exist('moviename','var') && ~isfield(trx,'moviename'),
    for i = 1:length(trx),
      trx(i).moviename = moviename;
    end
  end
  if dosave && ~isempty(savename),
      [didcopy,msg,~] = copyfile(matname,savename);
      if ~didcopy,
          error('Could not copy %s to %s:\n%s',matname,savename,msg);
      end
  end
      
end

% member functions can be weird
if verbose,
  fprintf('Adding off\n');
end
for i = 1:length(trx),
  trx(i).off = -trx(i).firstframe + 1;
  trx(i).matname = matname;
end

% make everything column matrices
fns = fieldnames(trx);
for i = 1:numel(fns),
  % don't do this for things that aren't numeric
  if any(~cellfun(@(x) isnumeric(x),{trx.(fns{i})})),
    continue;
  end
  % don't do this for things that have more than one dimension
  if any(cellfun(@(x) nnz(size(x)>1)>1,{trx.(fns{i})}));
    continue;
  end
  for j = 1:numel(trx),
    trx(j).(fns{i}) = trx(j).(fns{i})(:)';
  end
end

succeeded = true;
if verbose,
  fprintf('Done. returning from load_tracks\n');
end
function createFromDeeplabcut(dlc_data_file,mov_dir)
% function convertFromDeeplabcut(dlc_data_file,mov_dir)
% Creates a APT project from Deeplabcut project.
% dlc_data_file is the csv file generated by deeplabcut during training.
% mov_dir is the directory with the movie files used for the deeplabcut
% project. If not specified, script will ask the user for location of each movie file.

%mov_dir = myparse(varargin,'mov_dir','');
if nargin < 2,
  mov_dir = '';
end

fid = fopen(dlc_data_file,'r');
A = textscan(fid,'%s');
A = A{1};
fclose(fid);

bparts = split(A{2},',');
sc_name = bparts{1};
bparts = bparts(2:2:end);
npts = numel(bparts);
movies = {};
frames = [];
pts = {};
movies_short = {};
movies_long = {};
for ndx = 4:numel(A)
  ss = split(A{ndx},',');
  dd = split(ss{1},'/');
  mov_name = dd{2};
  mndx = find(strcmp(mov_name,movies_short));
  if isempty(mndx)
    movie_found = false;
    if ~isempty(mov_dir)
      ddir = dir(fullfile(mov_dir,[mov_name '*']));
      if numel(ddir)==1      
        cur_mov = fullfile(ddir(1).folder,ddir(1).name);
        movie_found = true;
      end
    end
    if ~movie_found
      m_str = sprintf('Locate %s',mov_name);
      [ffile,fdir] = uigetfile('*.avi',m_str);
      if ffile==0,
        errordlg(sprintf('Could not find movie %s',mov_name));
        return;
      end
      cur_mov = fullfile(fdir,ffile);
    end
    movies_short{end+1} = mov_name;
    movies_long{end+1} = cur_mov;
  else
    cur_mov = movies_long{mndx};
  end
  movies{end+1} = cur_mov;
  ff = splitext(dd{3});
  cur_f = str2double(ff(4:end));
  frames(end+1) = cur_f;
  cur_pt = [];
  for pndx = 2:numel(ss)
    cur_pt(end+1) = str2double(ss{pndx});
  end
  pts{end+1,1} = cur_pt;
end
pts = cell2mat(pts);
pts = reshape(pts,[],2,npts);
pts = permute(pts,[1 3 2]);

[umovies,~,movieidx] = unique(movies);

%%

cfgBase = Labeler.cfgGetLastProjectConfigNoView;
% cfgBase = ReadYaml(Labeler.DEFAULT_CFG_FILENAME);
cfgBase.NumLabelPoints = npts;
cfgBase.LabelPointNames = bparts;
cfgBase.ProjectName = sc_name;

cfgBase.NumViews = 1;
cfgBase.Trx.HasTrx = false;
cfgBase.ViewNames = {};
cfgBase.Track.Enable = true;
cfgBase.ProjectName = 'test';
FIELDS2DOUBLIFY = {'Gamma' 'FigurePos' 'AxisLim' 'InvertMovie' 'AxFontSize' 'ShowAxTicks' 'ShowGrid'};
cfgBase.View(1) = ProjectSetup('structLeavesStr2Double',cfgBase.View(1),FIELDS2DOUBLIFY);

lObj = StartAPT();
lObj.initFromConfig(cfgBase);
lObj.projNew(cfgBase.ProjectName);

for ndx = 1:numel(umovies)
  lObj.movieAdd(umovies(ndx));
end

lObj.movieSet(1);
%%

for ndx = 1:numel(movieidx)
  lObj.labeledpos{movieidx(ndx)}(:,:,frames(ndx)) = pts(ndx,:,:);
end

s = lObj.projGetSaveStruct();
[success,lblfname] = lObj.projSaveSmart();
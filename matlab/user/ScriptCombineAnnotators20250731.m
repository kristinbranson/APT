%% set parameters

% output directory
outdir = 'tmpdata';

% directory containing all the lbl files to combine
lbldir = '/groups/branson/bransonlab/aniket/APT/label_files/raw_label_files_annotators';

% lblfile to start with -- should have all the movies in it
baselblfile = 'walshjTest.lbl';

% lblfiles to combine, in order -- later will overwrite earlier
addlblfiles = {
  'Nmanakov32.lbl'
  'Nmanakov40.lbl'
  'Nmanakov41.lbl'
  'Nmanakov48.lbl'
  'Nmanakov49.lbl'
  'Nmanakov50.lbl'
  'Nmanakov51.lbl'
  'Nmanakov52.lbl'
  'Nmanakov53.lbl'
  'Nmanakov54.lbl'
  'Nmanakov55.lbl'
  'Nmanakov56.lbl'
  'Nmanakov57.lbl'
  'Nmanakov58.lbl'
  'Nmanakov60.lbl'
  'walshjTest.lbl' % this is the same as base, but i want to overwrite everything with base
  };

% location of old calibration information
old_data_dir = '/groups/branson/bransonlab/PTR_fly_annotations_3D/';

% location of newly cropped videos
newcroppedmoviedir = '/groups/branson/bransonlab/aniket/APT/3D_labeling_project/movie_output_dir_combined_views';

% fixed size of the two cropped videos
recropped_image_sizes = [1331, 1167]; % [virtual image, real image]

% indices of cameras in movie names
cams = [0,1];

% indices of view types
bottom_views = [1,3];
side_views = [2,4];

% experiments with ufmf compression artifacts to remove
badexps = [39,52,53];

% where to output combined table for python manipulation
outmatfile = fullfile(outdir,'combined_table.mat');

% where to save the result
outlblfile_combined = fullfile(outdir,'combined.lbl');

%% open the base file

inlblfile_combined = fullfile(lbldir,baselblfile);
lObj = StartAPT;
lObj.projLoadGUI(inlblfile_combined);
moviefiles_base = lObj.movieFilesAllFull;
moviefiles_gt_base = lObj.movieFilesAllGTFull;
tbldata_base = lObj.labelGetMFTableLabeled('useMovNames',true);

%% collect all labels from all lbl files

newmoviefiles = cell(0,size(moviefiles_base,2));
newmoviefiles_gt = cell(0,size(moviefiles_base,2));
tbldata = cell(size(addlblfiles));
for i = 1:numel(addlblfiles),
  inlblfile = fullfile(lbldir,addlblfiles{i});
  lObj.projLoadGUI(inlblfile);
  moviefilescurr = lObj.movieFilesAllFull;
  % check that all movies are in the base file
  isnew = ~ismember(moviefilescurr(:,1),moviefiles_base(:,1));
  if any(isnew),
    fprintf('%s movies found in lblfile %d %s not in base file\n',nnz(isnew),i,addlblfiles{i});
    fprintf('%s\n',moviefilescurr{isnew,1});
    newmoviefiles(end+1:end+nnz(isnew),:) = moviefilescurr(isnew,:);
  end
  moviefilescurr = lObj.movieFilesAllGTFull;
  isnew = ~ismember(moviefilescurr(:,1),moviefiles_gt_base(:,1));
  if any(isnew),
    fprintf('%s GT movies found in lblfile %d %s not in base file\n',nnz(isnew),i,addlblfiles{i});
    fprintf('%s\n',moviefilescurr{isnew,1});
    newmoviefiles_gt(end+1:end+nnz(isnew),:) = moviefilescurr(isnew,:);
  end
  tbldata{i} = lObj.labelGetMFTableLabeled('useMovNames',true);
end

% reload the base lbl file
lObj.projLoadGUI(inlblfile_combined);

tbldata_combined = tbldata_base;
for i = 1:numel(addlblfiles),
  ism = tblismember(tbldata{i},tbldata_combined,MFTable.FLDSID);
  overwrite = tblismember(tbldata_combined,tbldata{i},MFTable.FLDSID);
  fprintf('Lblfile %d, %d overwritten labels, %d new labels\n',i,nnz(overwrite),nnz(~ism));
  tbldata_combined = tblvertcatsafe(tbldata_combined(~overwrite,:),tbldata{i});
end

assert(isempty(newmoviefiles));

% Lblfile 1, 17 overwritten labels, 45 new labels
% Lblfile 2, 24 overwritten labels, 3 new labels
% Lblfile 3, 56 overwritten labels, 0 new labels
% Lblfile 4, 27 overwritten labels, 0 new labels
% Lblfile 5, 59 overwritten labels, 6 new labels
% Lblfile 6, 59 overwritten labels, 11 new labels
% Lblfile 7, 66 overwritten labels, 10 new labels
% Lblfile 8, 76 overwritten labels, 6 new labels
% Lblfile 9, 59 overwritten labels, 16 new labels
% Lblfile 10, 70 overwritten labels, 7 new labels
% Lblfile 11, 77 overwritten labels, 7 new labels
% Lblfile 12, 67 overwritten labels, 0 new labels
% Lblfile 13, 75 overwritten labels, 10 new labels
% Lblfile 14, 59 overwritten labels, 6 new labels
% Lblfile 15, 68 overwritten labels, 14 new labels
% Lblfile 16, 157 overwritten labels, 0 new labels

%% remove all movies that don't have labels and bad movies

[~,movidx] = unique(tbldata_combined.mov(:,1));
moviefiles_labeled = tbldata_combined.mov(movidx,:);

isunlabeled = ~ismember(moviefiles_base(:,1),moviefiles_labeled(:,1));
exps_base = regexp(moviefiles_base(:,1),'exp_?(\d+)/','tokens','once');
exps_base = cellfun(@str2num,[exps_base{:}]);
assert(numel(exps_base)==size(moviefiles_base,1));
toremove = isunlabeled | ismember(exps_base,badexps)';
moviefiles_remove = moviefiles_base(toremove,:);

for i = 1:size(moviefiles_remove,1),
  moviefile = moviefiles_remove{i,1};
  imov = find(strcmp(moviefile,lObj.movieFilesAllFull(:,1)));
  assert(numel(imov) == 1);
  tfSucc = lObj.movieRmGUI(imov,'force',true);
  assert(tfSucc);
end

% save to output
lObj.projSave(outlblfile_combined);
moviefiles_base = lObj.movieFilesAllFull;

%% for python manipulation
% output combined label file if you need to do some python stuff with it

% convert to struct
s_combined = labelTable2struct(tbldata_combined);
% output to hdf5 file -- can mess around with in python
save(outmatfile,'-struct','s_combined');

% do some stuff in python
% ...

% to load in outputs from python
% s_combined = load(outmatfile);
% tbldata_combined = struct2table(s_combined);

%% find new cropped versions of the movie

newmoviefiles_base = cell(size(moviefiles_base));

for i = 1:size(moviefiles_base,1),
  moviefilescurr = moviefiles_base(i,:);
  for j = 1:numel(cams),
    camcurr = cams(j);
    idxcam = find(~cellfun(@isempty,regexp(moviefilescurr,sprintf('cam_%d',camcurr),'once')));
    assert(numel(idxcam)==2,'Could not match cam');
    m = regexp(moviefilescurr(idxcam),'exp(?<exp>\d+)/movies\d?/(?<mov>.*)_crop_col(?<col>\d+)to','once','names');
    m = [m{:}];
    assert(numel(m)==2);
    oldcols = cellfun(@str2num,{m.col});
    [~,oldorder] = sort(oldcols);
    searchstr = fullfile(newcroppedmoviedir,['exp_',m(1).exp],[m(1).mov,'*ufmf']);
    newmoviefiles = mydir(searchstr);
    assert(numel(newmoviefiles)==2,'Could not find new movies');

    % sort by first column
    newcols = regexp(newmoviefiles,'_crop_col(\d+)to','tokens','once');
    newcols = cellfun(@str2num,[newcols{:}]);
    [~,neworder] = sort(newcols);
    newmoviefiles_base(i,idxcam(oldorder)) = newmoviefiles(neworder);
  end
end

%% replace movie paths in project

assert(isequal(size(lObj.movieFilesAll),size(newmoviefiles_base)));
lObj.movieFilesAll = newmoviefiles_base;
lObj.projSave(outlblfile_combined);

%% replace movie paths in label table

[ism,movidx] = ismember(tbldata_combined.mov(:,1),moviefiles_base(:,1));

% remove labels for which we removeed videos
tbldata_combined = tbldata_combined(ism,:);
movidx = movidx(ism);
tbldata_combined.mov = newmoviefiles_base(movidx,:);

%% load in new cropping coordinates

experiment_dirs = dir(fullfile(old_data_dir, 'exp*'));
offset_virtual_cam = zeros(numel(experiment_dirs),2); % Offset (pixels) to add to the real cam coordinates to correct cropping for the virtual view of each camera
cropexpnums = nan(numel(experiment_dirs),1);

for i = 1:length(experiment_dirs)
  calibration_file_path = dir(fullfile(old_data_dir, experiment_dirs(i).name, '*calibration_data.mat'));
  if isempty(calibration_file_path)
    disp(['Calibration file not found for exp ', i])
  end
  calibration_data = load(fullfile(calibration_file_path.folder, calibration_file_path.name));
  dividing_col = calibration_data.dividing_col;
  offset_virtual_cam(i,1) = recropped_image_sizes(1) - dividing_col(1);
  offset_virtual_cam(i,2) = recropped_image_sizes(1) - dividing_col(2);
  m = regexp(experiment_dirs(i).name,'exp_?(\d+)$','tokens','once');
  assert(~isempty(m));
  cropexpnums(i) = str2double(m{1});
end

%% fix coordinates for cropping

m = regexp(newmoviefiles_base(:,1),'exp_?(\d+)/','tokens','once');
projexpnums = cellfun(@(x) str2double(x),[m{:}])';
nviews = lObj.nview;
nkpts = lObj.nPhysPoints;

for i = 1:size(newmoviefiles_base,1),
  idxcurr = movidx == i;
  expnumcurr = projexpnums(i);
  cropi = find(expnumcurr==cropexpnums);
  offset_curr = offset_virtual_cam(cropi,:);
  pcurr = tbldata_combined.p(idxcurr,:);
  % labeled keypoints are stored in the order
  % nlabels x nkpts x nviews x 2
  nlabelscurr = size(pcurr,1);
  pcurr = reshape(pcurr,[nlabelscurr,nkpts,nviews,2]);
  % add to the y-coordinate of bottom views
  for j = 1:numel(bottom_views),
    pcurr(:,:,bottom_views(j),2) = pcurr(:,:,bottom_views(j),2) + offset_curr(j);
  end
  tbldata_combined.p(idxcurr,:) = reshape(pcurr,nlabelscurr,[]);
end

%% import label table

lObj.labelPosBulkImportTbl(tbldata_combined);

%% save result

lObj.projSave(outlblfile_combined);

%% output images of all labels

outimgdir = fullfile(outdir,'labeled_images');
if ~exist(outimgdir,'dir'),
  mkdir(outimgdir);
end
colors = lObj.labelPointsPlotInfo.Colors;
hfig = figure(9);
set(hfig,'Position',[270,1100,3000,800]);
clf;
htile = tiledlayout(1,nviews,'TileSpacing','none','Padding','none');
hax = gobjects(1,nviews);
tbldata_out = lObj.labelGetMFTableLabeled('useMovNames',true);
for i = 1:nviews,
  hax(i) = nexttile;
end
border = 40; % pixels
set(hax,'XTick',[],'YTick',[]);
for i = 1:size(lObj.movieFilesAllFull,1),
  moviefilescurr = lObj.movieFilesAllFull(i,:);
  idxcurr = find(strcmp(moviefilescurr{1},tbldata_out.mov(:,1)));
  readframes = cell(1,nviews);
  fids = cell(1,nviews);
  for j = 1:nviews,
    [readframes{j},~,fids{j}] = get_readframe_fcn(moviefilescurr{j});
  end
  for exi = idxcurr(:)',
    fr = tbldata_out.frm(exi);
    pcurr = tbldata_out.p(exi,:);
    pcurr = reshape(pcurr,[nkpts,nviews,2]);
    mincoord = permute(min(pcurr,[],1),[2,3,1]);
    maxcoord = permute(max(pcurr,[],1),[2,3,1]);

    for view = 1:nviews,
      im = readframes{view}(fr);
      cla(hax(view));
      image(hax(view),im);
      colormap(hax(view),'gray');
      hold(hax(view),'on');
      for kpt = 1:nkpts,
        plot(hax(view),pcurr(kpt,view,1),pcurr(kpt,view,2),'.','Color',colors(kpt,:),'MarkerSize',12);
      end
      axis(hax(view),'image','off');
      xlim = [mincoord(view,1),maxcoord(view,1)]+border*[-1,1];
      ylim = [mincoord(view,2),maxcoord(view,2)]+border*[-1,1];
      set(hax(view),'XLim',xlim,'YLim',ylim);
    end
    expnum = regexp(moviefilescurr{1},'exp_?(\d+)/','once','tokens');
    ti = sprintf(' ex %d, movie set %d, exp %s, frame %d',exi,i,expnum{1},fr);
    text(hax(1),mincoord(1,1)-border,mincoord(1,2)-border+5,ti,'HorizontalAlignment','left','VerticalAlignment','top','Color','m');
    outfile = fullfile(outimgdir,sprintf('example%03d_movieset%02d_exp%s_fr%06d.png',exi,i,expnum{1},fr));
    saveas(hfig,outfile,'png');
  end
  for j = 1:nviews,
    if ~isempty(fids{j}) && fids{j} > 0,
      fclose(fids{j});
    end
  end
end
fprintf('Done\n');


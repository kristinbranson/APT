% which views correspond to bottom/side
bottom_views = [1,3];
side_views = [2,4];

% open input project
lblfile_mv = '/groups/branson/bransonlab/aniket/APT/label_files/recroppedFourViews_trained.lbl';
lObj_mv = StartAPT;
lObj_mv.projLoadGUI(lblfile_mv);

% export labels to table
tbl_mv = lObj_mv.labelGetMFTableLabeled('useMovNames',true);

% labeled keypoints are stored in the order
% nlabels x nkpts x nviews x 2
nlabels = size(tbl_mv,1);
nkpts = lObj_mv.nPhysPoints;
nviews = lObj_mv.nview;
preshape = reshape(tbl_mv.p,[nlabels,nkpts,nviews,2]);
tfocc_reshape = reshape(tbl_mv.tfocc,[nlabels,nkpts,nviews]);

% close the mv project
close(lObj_mv.controller_.mainFigure_);

%% bottom view

% make new table with just the bottom views
for i = 1:numel(bottom_views),
  tbl_curr = tbl_mv;
  tbl_curr.mov = tbl_mv.mov(:,bottom_views(i));
  % 1 here is because multitarget
  tbl_curr.p = reshape(preshape(:,:,bottom_views(i),:),[nlabels,1,nkpts*2]);
  tbl_curr.tfocc = reshape(tfocc_reshape(:,:,bottom_views(i)),[nlabels,1,nkpts]);
  if i == 1,
    tbl_bottom = tbl_curr;
  else
    tbl_bottom = [tbl_bottom;tbl_curr];
  end
end


% open the bottom view project
lblfile_bottom = '/groups/branson/bransonlab/apt/unittest/flyprism/combinedBottomViewMA20250801.lbl';
outlblfile_bottom = '/groups/branson/bransonlab/apt/unittest/flyprism/combinedBottomViewMA20250825.lbl';
lObj_bottom = StartAPT;
lObj_bottom.projLoadGUI(lblfile_bottom);

% add missing movies
allmovs = tbl_bottom.mov;
oldmovs = lObj_bottom.movieFilesAllFull;
newmovs = setdiff(allmovs,oldmovs);

for i = 1:numel(newmovs),
  lObj_bottom.movieAdd(newmovs{i},'','offerMacroization',false);
end

% import label table
lObj_bottom.labelPosBulkImportTbl(tbl_bottom);

% save project to new name
lObj_bottom.projSave(outlblfile_bottom) ;

close(lObj_bottom.controller_.mainFigure_);


%% side view

% make new table with just the side view
for i = 1:numel(side_views),
  tbl_curr = tbl_mv;
  tbl_curr.mov = tbl_mv.mov(:,side_views(i));
  % 1 here is because multitarget
  tbl_curr.p = reshape(preshape(:,:,side_views(i),:),[nlabels,1,nkpts*2]);
  tbl_curr.tfocc = reshape(tfocc_reshape(:,:,side_views(i)),[nlabels,1,nkpts]);
  if i == 1,
    tbl_side = tbl_curr;
  else
    tbl_side = [tbl_side;tbl_curr];
  end
end


% open the side view project
lblfile_side = '/groups/branson/bransonlab/apt/unittest/flyprism/combinedSideViewMA20250801.lbl';
outlblfile_side = '/groups/branson/bransonlab/apt/unittest/flyprism/combinedSideViewMA20250825.lbl';
lObj_side = StartAPT;
lObj_side.projLoadGUI(lblfile_side);

% add missing movies
allmovs = tbl_side.mov;
oldmovs = lObj_side.movieFilesAllFull;
newmovs = setdiff(allmovs,oldmovs);

for i = 1:numel(newmovs),
  lObj_side.movieAdd(newmovs{i},'','offerMacroization',false);
end

% import label table
lObj_side.labelPosBulkImportTbl(tbl_side);

% save project to new name
lObj_side.projSave(outlblfile_side) ;

close(lObj_side.controller_.mainFigure_);

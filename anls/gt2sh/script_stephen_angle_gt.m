%% Stephen fly Angle vs Error 

%% MK 14032022

%% files

%bodylbl_files = '/groups/huston/hustonlab/flp-chrimson_experiments/fly2BodyAxis_lookupTable_Ben.csv';
gt_lbl_file = '/groups/branson/bransonlab/apt/experiments/data/sh_trn4992_gtcomplete_cacheddata_updated20200317_stripped_mdn.lbl';
gt_res_file = '/groups/branson/bransonlab/mayank/apt_results/stephen_deepnet_20200706_view0_time20200917.mat';
gt_res_file2 = '/groups/branson/bransonlab/mayank/apt_results/stephen_deepnet_20200706_view1_time20200917.mat';
FLYNUM2BA = '/groups/branson/bransonlab/mayank/APT/matlab/user/flynum2bodyAxis_linux.csv';
FLYNUM2DLT = '/groups/huston/hustonlab/flp-chrimson_experiments/fly2DLT_lookupTableStephen.csv';
trn_file = '/groups/branson/bransonlab/apt/experiments/data/SelectedGTFrames_SJH_20180603.mat';
pred_lbls = '/groups/branson/bransonlab/apt/experiments/data/fly2RefPredLbl20180724.mat';
%pred_lbls = '/groups/branson/bransonlab/apt/experiments/data/fly2RefPredLbl20180724_fixed.mat';

APT.setpathsmart;
addpath('matlab/user/APT2RT');
addpath('/groups/branson/bransonlab/mayank/APT/user/APT2RT/matGeom/matGeom/geom3d');
pt1 = 6; % neck
pt2 = 8; % abdomen
pred_net = 'mdn_unet';

%% load them stuff

J = load(gt_lbl_file,'-mat');
K = load(gt_res_file);
K2 = load(gt_res_file2);
Q = load(trn_file);
P = load(pred_lbls);
f2ba = readtable(FLYNUM2BA);
f2ba.Properties.VariableNames = {'fly' 'lbl' 'mov' 'frm' 'calib'};

%% for each gt mov find the crop and the body labels

ngt = size(K.(pred_net){end}.info,1);
info = {};
not_found = zeros(0,2);
fly2dlt = readtable(FLYNUM2DLT,'Delimiter',',');
fly2dlt.Properties.VariableNames = {'fly' 'calib'};
all_calib = {};
all_crops = {};
is_enrich = [];

enrich_ids = Q.allflyids(Q.enrichedflies);
pred_info = squeeze(K.(pred_net){end}.info);

% The path to some of the movies in the GT label file don't match the ones
% in the pred lbl files.
J.movieFilesAllGT{39,1} = '/groups/huston/hustonlab/flp-chrimson_experiments/fly_210_to_218_28_10_15_SS02323_x_norpACsChrimsonFlp11/fly216/fly216_trial5/C001H001S0001/C001H001S0001_c.avi';
J.movieFilesAllGT{39,2} = '/groups/huston/hustonlab/flp-chrimson_experiments/fly_210_to_218_28_10_15_SS02323_x_norpACsChrimsonFlp11/fly216/fly216_trial5/C002H001S0001/C002H001S0001_c.avi';
J.movieFilesAllGT{45,1} = '/groups/huston/hustonlab/flp-chrimson_experiments/fly_229_to_238_1st_to_2nd_12_15_norpAchrimsonFLP_SS002323/fly229_300ms_stim/fly229_trial10_125fps/C001H001S0001/C001H001S0001_c.avi';
J.movieFilesAllGT{45,2} = '/groups/huston/hustonlab/flp-chrimson_experiments/fly_229_to_238_1st_to_2nd_12_15_norpAchrimsonFLP_SS002323/fly229_300ms_stim/fly229_trial10_125fps/C002H001S0001/C002H001S0001_c.avi';
J.movieFilesAllGT{63,1} = '/groups/huston/hustonlab/flp-chrimson_experiments/fly_219_to_228_28_10_15_SS00325_x_norpAcsChrimsonFlp11/fly219/fly219_trial2/C001H001S0001/C001H001S0001_c.avi';
J.movieFilesAllGT{63,2} = '/groups/huston/hustonlab/flp-chrimson_experiments/fly_219_to_228_28_10_15_SS00325_x_norpAcsChrimsonFlp11/fly219/fly219_trial2/C002H001S0001/C002H001S0001_c.avi';
J.movieFilesAllGT{89,1} = '/groups/huston/hustonlab/flp-chrimson_experiments/fly_488_to_502_SS02323_and_SS00325_8_3_2017/fly492/C001H001S0007/C001H001S0007_c.avi';
J.movieFilesAllGT{89,2} = '/groups/huston/hustonlab/flp-chrimson_experiments/fly_488_to_502_SS02323_and_SS00325_8_3_2017/fly492/C002H001S0007/C002H001S0007_c.avi';

failed = {};
for ndx =1:size(J.movieFilesAllGT,1)
  
  %%
  
  % Find the flynumber for the current movie
  mov_files = J.movieFilesAllGT{ndx,1};
  mov_files = strrep(mov_files,'$flpCE','/groups/huston/hustonlab/flp-chrimson_experiments');
  mov_files = strrep(mov_files,'\','/');
  mov_files = strrep(mov_files,'_c_i.avi','_c.avi');
  [~,ss1] = fileparts(fileparts(fileparts(mov_files)));  
  hh = strsplit(ss1(4:end),'_');
  fly_num = str2double(hh{1});
  if isnan(fly_num)
    [~,ss1] = fileparts(fileparts(fileparts(fileparts(mov_files))));
    fly_num = str2double(ss1(4:end));    
  end
  
  % load the calibration data
  dndx = find(fly2dlt.fly==fly_num);
  calfile = fly2dlt.calib{dndx};
  cal2 = CalRig.loadCreateCalRigObjFromFile(calfile);  
  all_calib{end+1} = cal2;
  
  % Whether enriched or not
  cur_enrich = sum(enrich_ids==fly_num)>0.5;
  if cur_enrich
    is_enrich(end+1) = 1;
  else
    is_enrich(end+1) = 0;
  end

  % Find the prediction label file that matches the movie in the GT label
  % file. This is required because there can be multiple prediction file
  % for the same fly
  tfile_ptn = sprintf('/groups/huston/hustonlab/flp-chrimson_experiments/APT_projectFiles/fly%d*.lbl',fly_num);
  tfiles = dir(tfile_ptn);
  matched = false;
  for tndx = 1:numel(tfiles)
    tlbl_file = fullfile(tfiles(tndx).folder,tfiles(tndx).name);
    tlbl = loadLbl(tlbl_file);
    
    tlbl_mov = strrep(tlbl.movieFilesAll,'Z:\','/groups/huston/hustonlab/');
    tlbl_mov = strrep(tlbl_mov,'\','/');
    tlbl_mov = strrep(tlbl_mov,'_c_i.avi','_c.avi');
    tlbl_mov_ndx = find(strcmp(tlbl_mov(:,1),mov_files));
    
    if ~isempty(tlbl_mov_ndx)
      matched = true;
      break;
    end
  end
  
  % There is one failure case
  if ~matched,
    failed{end+1} = {ndx, mov_files, tlbl_mov{1,1}};
    continue
  end

  % activation interval
    activ_int = flyNum2stimFrames_SJH(fly_num);

  % Find the GT frames that have been labeled
  ll = SparseLabelArray.full(J.labeledposGT{ndx});
  frames = find(~isnan(ll(1,1,:)));
  
  init_pivot = [];
  [axisAngleDegXYZ,translations,residualErrors,scaleErrors,...
  quaternion,EulerAnglesDeg_XYZ,pivot,refHead,threeD_pos] = ...
    APT2RT_al(tlbl_file,FLYNUM2BA,FLYNUM2DLT,...
    1, init_pivot, [],'flyNum',fly_num);

  % Replace the prediction data with gt labeled data to get its angle
  tlbl2 = tlbl;
  preds_gt = SparseLabelArray.full(tlbl.labeledpos2{tlbl_mov_ndx});
  preds_pts = preds_gt(:,:,frames);
  labeled_pts = ll(:,:,frames);
  preds_gt(:,:,frames) = ll(:,:,frames);
  preds_gt = SparseLabelArray.create(preds_gt,'nan');
  tlbl2.labeledpos2{tlbl_mov_ndx} = preds_gt;
  [axisAngleDegXYZ_gt,~,~,~,...
  ~,~,~,~,threeD_pos_gt] = ...
    APT2RT_al(tlbl2,FLYNUM2BA,FLYNUM2DLT,...
    1, pivot, refHead,'flyNum',fly_num,'iMov0',tlbl_mov_ndx,'iMov1',tlbl_mov_ndx);

  preds_angle = axisAngleDegXYZ(frames,:,tlbl_mov_ndx);
  gt_angle = axisAngleDegXYZ_gt(frames,:);
  for lndx = 1:numel(frames)
    
    % Another way to compute error is to give the labeled frame's 3d
    % position as the ref head and then compute of the prediction to that.

    cur_refHead = squeeze(cell2array(cellfun(@(x) x(frames(lndx),:),threeD_pos_gt{tlbl_mov_ndx},'UniformOutput',false)))';
    [axisAngleDegXYZ_gt1,~,~,~,~,~,~,~,threeD_pos_gt1] = ...
      APT2RT_al(tlbl,FLYNUM2BA,FLYNUM2DLT,...
      1, pivot, cur_refHead,'flyNum',fly_num,'iMov0',tlbl_mov_ndx,'iMov1',tlbl_mov_ndx);
    
    gg = frames(lndx)>=activ_int(:,1) & frames(lndx)<=activ_int(:,2);
    is_activ = any(gg);
    info{end+1} = {ndx, frames(lndx), fly_num, is_activ, ...
            cur_enrich, preds_pts,labeled_pts, ...
            preds_angle(lndx,:), gt_angle(lndx,:), axisAngleDegXYZ_gt1(frames(lndx),:)
            };
    
  end
  
  
end

save('~/temp/stephen_analysis.mat','info');

%% Separate out the statistics

err = [];
err2 = [];
ang = [];
ang2 = [];
act = [];
enrich = [];
errxyz = [];
errxyz1 = [];
for ndx = 1:numel(info)
  enrich(ndx) = info{ndx}{5};
  act(ndx) = info{ndx}{4};
  err(ndx) = abs(info{ndx}{8}(4)-info{ndx}{9}(4));
  ang(ndx) = info{ndx}{9}(4);
  ang2(ndx) = info{ndx}{8}(4);
  err2(ndx) = info{ndx}{10}(4);
  errxyz(ndx,:) = info{ndx}{9}(1:3)-info{ndx}{8}(1:3);
  errxyz1(ndx,:) = info{ndx}{10}(1:3);
end

sel = err<50;
enrich = enrich(sel);
act = act(sel);
err = err(sel);
ang = ang(sel);
ang2 = ang2(sel);
err2 = err2(sel);
errxyz = errxyz(sel,:);
errxyz1 = errxyz1(sel,:);

fprintf('All labels -- err mean:%.2f err std:%.2f\n', mean(err2), std(err2));
fprintf('Activated labels -- err mean:%.2f err std:%.2f\n', mean(err2(act>0.5)), std(err2(act>0.5)));
fprintf('InActive labels -- err mean:%.2f err std:%.2f\n', mean(err2(act<0.5)), std(err2(act<0.5)));
fprintf('Enrich labels -- err mean:%.2f err std:%.2f\n', mean(err2(enrich>0.5)), std(err2(enrich>0.5)));
fprintf('Not Enrich -- err mean:%.2f err std:%.2f\n', mean(err2(enrich<0.5)), std(err2(enrich<0.5)));
fprintf('Angles all:%.2f, enrich:%.2f, not enrich:%.2f, act:%.2f inactive:%.2f\n',...
  mean(ang), mean(ang(enrich>0.5)), mean(ang(enrich<0.5)), ...
  mean(ang(act>0.5)), mean(ang(act<0.5)));

%% snippet for debug

% Run from <APT> dir
FLYNUM2BA = '/groups/branson/bransonlab/mayank/APT/matlab/user/flynum2bodyAxis_linux.csv';
FLYNUM2DLT = '/groups/huston/hustonlab/flp-chrimson_experiments/fly2DLT_lookupTableStephen.csv';
pred_lbl = '/groups/huston/hustonlab/flp-chrimson_experiments/APT_projectFiles/fly636_intensity_6.lbl';
fly_num = 636;

APT.setpathsmart;
addpath('matlab/user/APT2RT');
addpath('user/APT2RT/matGeom/matGeom/geom3d');

[axisAngleDegXYZ,translations,residualErrors,scaleErrors,...
quaternion,EulerAnglesDeg_XYZ,pivot,refHead] = ...
  APT2RT(pred_lbl,FLYNUM2BA,FLYNUM2DLT,...
  1, [], []);%,'flyNum',fly_num);

figure; hist(vectorize(axisAngleDegXYZ(:,4,:)));


%%

ngt = size(K.(pred_net){end}.info,1);
info = {};
not_found = zeros(0,2);
fly2dlt = readtable(FLYNUM2DLT,'Delimiter',',');
fly2dlt.Properties.VariableNames = {'fly' 'calib'};
all_calib = {};
all_crops = {};
is_enrich = [];

enrich_ids = Q.allflyids(Q.enrichedflies);
pred_info = squeeze(K.(pred_net){end}.info);

% The path to some of the movies in the GT label file don't match the ones
% in the pred lbl files.
J.movieFilesAllGT{39,1} = '/groups/huston/hustonlab/flp-chrimson_experiments/fly_210_to_218_28_10_15_SS02323_x_norpACsChrimsonFlp11/fly216/fly216_trial5/C001H001S0001/C001H001S0001_c.avi';
J.movieFilesAllGT{39,2} = '/groups/huston/hustonlab/flp-chrimson_experiments/fly_210_to_218_28_10_15_SS02323_x_norpACsChrimsonFlp11/fly216/fly216_trial5/C002H001S0001/C002H001S0001_c.avi';
J.movieFilesAllGT{45,1} = '/groups/huston/hustonlab/flp-chrimson_experiments/fly_229_to_238_1st_to_2nd_12_15_norpAchrimsonFLP_SS002323/fly229_300ms_stim/fly229_trial10_125fps/C001H001S0001/C001H001S0001_c.avi';
J.movieFilesAllGT{45,2} = '/groups/huston/hustonlab/flp-chrimson_experiments/fly_229_to_238_1st_to_2nd_12_15_norpAchrimsonFLP_SS002323/fly229_300ms_stim/fly229_trial10_125fps/C002H001S0001/C002H001S0001_c.avi';
J.movieFilesAllGT{63,1} = '/groups/huston/hustonlab/flp-chrimson_experiments/fly_219_to_228_28_10_15_SS00325_x_norpAcsChrimsonFlp11/fly219/fly219_trial2/C001H001S0001/C001H001S0001_c.avi';
J.movieFilesAllGT{63,2} = '/groups/huston/hustonlab/flp-chrimson_experiments/fly_219_to_228_28_10_15_SS00325_x_norpAcsChrimsonFlp11/fly219/fly219_trial2/C002H001S0001/C002H001S0001_c.avi';
J.movieFilesAllGT{89,1} = '/groups/huston/hustonlab/flp-chrimson_experiments/fly_488_to_502_SS02323_and_SS00325_8_3_2017/fly492/C001H001S0007/C001H001S0007_c.avi';
J.movieFilesAllGT{89,2} = '/groups/huston/hustonlab/flp-chrimson_experiments/fly_488_to_502_SS02323_and_SS00325_8_3_2017/fly492/C002H001S0007/C002H001S0007_c.avi';

failed = {};
data = {};

%%
for ndx =1:size(J.movieFilesAllGT,1)
  
  
  % Find the flynumber for the current movie
  mov_files = J.movieFilesAllGT{ndx,1};
  mov_files = strrep(mov_files,'$flpCE','/groups/huston/hustonlab/flp-chrimson_experiments');
  mov_files = strrep(mov_files,'\','/');
  mov_files = strrep(mov_files,'_c_i.avi','_c.avi');
  [~,ss1] = fileparts(fileparts(fileparts(mov_files)));  
  hh = strsplit(ss1(4:end),'_');
  fly_num = str2double(hh{1});
  if isnan(fly_num)
    [~,ss1] = fileparts(fileparts(fileparts(fileparts(mov_files))));
    fly_num = str2double(ss1(4:end));    
  end
  
  % load the calibration data
  dndx = find(fly2dlt.fly==fly_num);
  calfile = fly2dlt.calib{dndx};
  cal2 = CalRig.loadCreateCalRigObjFromFile(calfile);  
  all_calib{end+1} = cal2;
  
  % Whether enriched or not
  cur_enrich = sum(enrich_ids==fly_num)>0.5;
  if cur_enrich
    is_enrich(end+1) = 1;
  else
    is_enrich(end+1) = 0;
  end

  % Find the prediction label file that matches the movie in the GT label
  % file. This is required because there can be multiple prediction file
  % for the same fly
  tfile_ptn = sprintf('/groups/huston/hustonlab/flp-chrimson_experiments/APT_projectFiles/fly%d*.lbl',fly_num);
  tfiles = dir(tfile_ptn);
  matched = false;
  for tndx = 1:numel(tfiles)
    tlbl_file = fullfile(tfiles(tndx).folder,tfiles(tndx).name);
    tlbl = loadLbl(tlbl_file);
    
    tlbl_mov = strrep(tlbl.movieFilesAll,'Z:\','/groups/huston/hustonlab/');
    tlbl_mov = strrep(tlbl_mov,'\','/');
    tlbl_mov = strrep(tlbl_mov,'_c_i.avi','_c.avi');
    tlbl_mov_ndx = find(strcmp(tlbl_mov(:,1),mov_files));
    
    if ~isempty(tlbl_mov_ndx)
      matched = true;
      break;
    end
  end
  
  % There is one failure case
  if ~matched
    failed{end+1} = {ndx, mov_files};
    continue
  end

  
  % activation interval
  activ_int = flyNum2stimFrames_SJH(fly_num);

  % Find the GT frames that have been labeled
  gt_labels = SparseLabelArray.full(J.labeledposGT{ndx});
  frames = find(~isnan(gt_labels(1,1,:)));
  
  
  max_f = 0;
  for tndx = 1:numel(tlbl.labeledpos2)
    max_f = max(tlbl.labeledpos2{tndx}.size(3),max_f);
  end
  
  kk = nan(10,2,max_f,numel(tlbl.labeledpos2));
  ll = nan(10,2,max_f,numel(tlbl.labeledpos2));
  
  for tndx = 1:numel(tlbl.labeledpos2)
    cursz = tlbl.labeledpos2{tndx}.size(3);
    kk(:,:,1:cursz,tndx) = SparseLabelArray.full(tlbl.labeledpos2{tndx});
    if tndx==tlbl_mov_ndx
      ll(:,:,1:cursz,tndx) = gt_labels;
    end
  end
  
  data{ndx} = {ndx, fly_num, kk,ll, mov_files, cal2, cur_enrich, tlbl_file};
      
end


%%

save('~/temp/stephen_GT_data.mat','data','failed');

%%
fly_nums = [];
for ndx =1:size(J.movieFilesAllGT,1)
  
  
  
  % Find the flynumber for the current movie
  mov_files = J.movieFilesAllGT{ndx,1};
  mov_files = strrep(mov_files,'$flpCE','/groups/huston/hustonlab/flp-chrimson_experiments');
  mov_files = strrep(mov_files,'\','/');
  mov_files = strrep(mov_files,'_c_i.avi','_c.avi');
  [~,ss1] = fileparts(fileparts(fileparts(mov_files)));  
  hh = strsplit(ss1(4:end),'_');
  fly_num = str2double(hh{1});
  if isnan(fly_num)
    [~,ss1] = fileparts(fileparts(fileparts(fileparts(mov_files))));
    fly_num = str2double(ss1(4:end));    
  end
  fly_nums(ndx) = fly_num;
end

%%
ufly_id = unique(fly_nums);
fly_count = zeros(1,numel(ufly_id));
for ndx =1:numel(ufly_id)
  fly_count(ndx) = nnz(fly_nums==ufly_id(ndx));
  if fly_count(ndx)>1
    ix = find(fly_nums==ufly_id(ndx));
    fprintf(' ----\n',fly_count(ndx));
    for xx = ix(:)'
      ll = SparseLabelArray.full(J.labeledposGT{xx});
      frames = find(~isnan(ll(1,1,:)));
      fprintf('%d %s\n',xx,J.movieFilesAllGT{xx,1});
      fprintf('%d \n',frames);
    end

  end
end


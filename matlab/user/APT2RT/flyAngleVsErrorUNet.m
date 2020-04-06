% training data
X = load('/groups/branson/bransonlab/mayank/PoseTF/headTracking/trnDataSH_20180503_notable.mat');
% list of label files
Y = load('/groups/branson/bransonlab/apt/experiments/data/cprXVerrVsHeadPosn20180529.mat');
% Add fly 708_intensity 6.
% Y.tFliesTrk2018.lbl{53} = strrep(Y.tFliesTrk2018.lbl{53},'intensity_3','intensity_6');
% Y.tFliesTrk2018.lbl{54} = strrep(Y.tFliesTrk2018.lbl{54},'intensity_3','intensity_4');
% Y.tFliesTrk2018.lbl{55} = strrep(Y.tFliesTrk2018.lbl{55},'intensity_3','intensity_6');
% Y.tFliesTrk2018.lbl{56} = strrep(Y.tFliesTrk2018.lbl{56},'intensity_3','intensity_6');
% CV predictions
P = load('/groups/branson/home/kabram/bransonlab/PoseTF/headTracking/normal_cv_data.mat');

outdir = '/groups/branson/home/kabram/temp/stephenOut/';
% 20191009 AL updated paths to matlab code
bodylut = '/groups/branson/home/kabram/bransonlab/APT/matlab/user/flynum2bodyAxis.csv';
mybodylut = '/groups/branson/home/kabram/bransonlab/APT/matlab/user/flynum2bodyAxis_linux.csv';
caliblut = '/groups/huston/hustonlab/flp-chrimson_experiments/fly2DLT_lookupTableStephen.csv';

addpath user/APT2RT/
APT.setpath
%% Combine all the lbl files for a fly into 1.


flyId_all = X.tMain.flyID;
uu = unique(flyId_all);
lbl_dir = '/groups/huston/hustonlab/flp-chrimson_experiments/APT_projectFiles/';

Y = struct;
Y.tFliesTrk2018.fly = [];
Y.tFliesTrk2018.lbl = {};
fields2join = {'movieFilesAll','trxFilesAll','movieInfoAll','labeledpos','labeledpostag',...
  'labeledposTS','labeledposMarked','labeledpos2'};
for curfly = uu(:)'
  if curfly < 100, 
    continue;
  end
  dd = dir(sprintf('%s/fly%d*.lbl',lbl_dir, curfly));
  for ndx = 1:numel(dd)
    zz = load(fullfile(lbl_dir,dd(ndx).name),'-mat');
    if ndx == 1,
      allzz = zz;
    else
      for fndx = 1:numel(fields2join),
        allzz.(fields2join{fndx}) = [allzz.(fields2join{fndx}); zz.(fields2join{fndx})];
      end
    end
  end
  joined_lbl = fullfile(outdir,sprintf('fly%d_joined.lbl',curfly));
  save(joined_lbl,'-struct','allzz');
  Y.tFliesTrk2018.fly(end+1) = curfly;
  Y.tFliesTrk2018.lbl{end+1} = joined_lbl;
end




%%
fid = fopen(bodylut,'r');
fido = fopen(mybodylut,'w');
l = fgetl(fid);
while ischar(l)
  l = strrep(l,'\','/');
  l = strrep(l,'Z:','/groups/huston/hustonlab');
  fprintf(fido,l);
  fprintf(fido,'\n');
  l = fgetl(fid);
end


%%


rmpath /groups/branson/bransonlab/mayank/tools/TOOLBOX_calib/
addpath /groups/branson/bransonlab/mayank/APT/matlab/user/APT2RT/matGeom/matGeom/geom3d

flyId_all = X.tMain.flyID;

split_type = 'easy';
pred_info_field = sprintf('info_%s_side',split_type);
pred_field_side = sprintf('pred_%s_side',split_type);
pred_field_front = sprintf('pred_%s_front',split_type);

info = P.(pred_info_field);
info(:,1) = info(:,1) + 1; %python to matlab indexing
info(:,2) = info(:,2) + 1;

l_angle = nan(size(info,1),4);
p_angle = nan(size(info,1),4);

for ndx = 1:numel(Y.tFliesTrk2018.fly)
  %%
  curfly = Y.tFliesTrk2018.fly(ndx);
  lbl_file = Y.tFliesTrk2018.lbl{ndx};
  lbl_file = strrep(lbl_file,'\','/');
  lbl_file = strrep(lbl_file,'Z:','/groups/huston/hustonlab');
  Z = load(lbl_file,'-mat');
  
  fly_data_ndx = find(flyId_all == curfly);
  
  new_labeled_pos = cell(size(Z.labeledpos));
  for lndx = 1:numel(Z.labeledpos)
    new_labeled_pos{lndx} = nan(Z.labeledpos{lndx}.size);
  end
  
  curinfo = struct;
  curinfo.pred_ndx = [];
  curinfo.lbl_mov_ndx = [];
  curinfo.ts = [];
  
  for fndx = fly_data_ndx(:)'
    curmovid = X.mov_id(fndx);
    ts = X.frm(fndx);
    
    % Find in the index in the prediction matrix
    pred_ndx = find(info(:,1)== curmovid & info(:,2) == ts);
    if numel(pred_ndx) ~= 1,
      fprintf('Couldnt find prediction for flynum:%d flyndx:%d movid:%d frame:%d\n',...
        curfly,fndx,curmovid,ts);
      continue;
    end
    
    % Find the index in the lbl file.
    cur_mov_orig = X.tMain.movFile{fndx,1};
    if curfly == 254,
      cur_mov_orig = strrep(cur_mov_orig,'fly253','fly254');
    end
    
    cur_mov = strrep(cur_mov_orig,'/','\');
    zz = strsplit(cur_mov,'\');
    zz = zz(end-2:end-1);
    lbl_mov_ndx = [];
    for idx = 1:size(Z.movieFilesAll,1)
      lbl_mov = Z.movieFilesAll{idx,1};
      lbl_mov = strrep(lbl_mov,'/','\');
      pp = strsplit(lbl_mov,'\');
      pp = pp(end-2:end-1);
      if isequal(zz,pp),
        lbl_mov_ndx(end+1) = idx;
      end
    end
    
%     lbl_mov_ndx = find(strcmp(Z.movieFilesAll(:,1),cur_mov));
    if numel(lbl_mov_ndx) ~= 1,
      fprintf('Couldnt find movie for flynum:%d %s\n',...
        curfly,cur_mov);
      continue;
    end
    curinfo.lbl_mov_ndx(end+1) = lbl_mov_ndx;
    curinfo.ts(end+1) = ts;
    curinfo.pred_ndx(end+1) = pred_ndx;
    
    new_labeled_pos{lbl_mov_ndx}(1:5,1,ts) = squeeze(P.(pred_field_side)(pred_ndx,:,1))+ ...
      1 + X.roi_crop2(fndx,1,1);
    new_labeled_pos{lbl_mov_ndx}(1:5,2,ts) = squeeze(P.(pred_field_side)(pred_ndx,:,2))+ ...
      1 + X.roi_crop2(fndx,3,1);
    new_labeled_pos{lbl_mov_ndx}(6:10,1,ts) = squeeze(P.(pred_field_front)(pred_ndx,:,1))+...
      1 + X.roi_crop2(fndx,1,2);
    new_labeled_pos{lbl_mov_ndx}(6:10,2,ts) = squeeze(P.(pred_field_front)(pred_ndx,:,2))+...
      1 + X.roi_crop2(fndx,3,2);
    
  end
  
  for lndx = 1:numel(Z.labeledpos)
%     Z.labeledpos{lndx} = SparseLabelArray.create(new_labeled_pos{lndx},'nan');
    Z.labeledpos{lndx} = new_labeled_pos{lndx};
    for view = 1:2,
      mov_file = strrep(Z.movieFilesAll{lndx,view},'\','/');
      mov_file = strrep(mov_file,'Z:','/groups/huston/hustonlab');
      mov_file = strrep(mov_file,'W:','/groups/huston/hustonlab');
      Z.movieFilesAll{lndx,view} = mov_file;
    end
  end
  
  
  [~,lbl_file_name,~] = fileparts(lbl_file);
  out_file = fullfile(outdir,[lbl_file_name '_updated.lbl']);
  save(out_file,'-struct','Z');
  %%
  try 
    [ang, tr, res, se, qt, piv, refH] = APT2RT(out_file, mybodylut, caliblut, 1, [], []);
    [ang, tr, res, se, qt, piv, refH] = APT2RT(out_file, mybodylut, caliblut, 1, piv, refH);
    [ang, tr, res, se, qt, piv, refH] = APT2RT(out_file, mybodylut, caliblut, 0, piv, refH);
  catch Me
    if strcmp(Me.identifier, 'ERR:missingBody')
      continue;
    else
      rethrow(Me)
    end
  end
    
  jj = zeros(numel(curinfo.ts),4);
  for idx = 1:numel(curinfo.ts)
    l_angle(curinfo.pred_ndx(idx),:) = ang(curinfo.ts(idx),:,curinfo.lbl_mov_ndx(idx));
    p_angle(curinfo.pred_ndx(idx),:) = ang(curinfo.ts(idx),:,curinfo.lbl_mov_ndx(idx));
    jj(idx,:) = ang(curinfo.ts(idx),:,curinfo.lbl_mov_ndx(idx));
  end
  
end


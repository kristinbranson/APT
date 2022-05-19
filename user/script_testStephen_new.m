lbl_file = '/groups/branson/home/kabram/APT_projects/sh_trn4992_gtcomplete_cacheddata_updated20200317_compress20200325_grone20220511.lbl';
tlbl_dir = tempname;
untar(lbl_file,tlbl_dir);
ndir = '/nearline/huston/flp-chrimson_experiments_pre_2020_videos/';
gdir = '/groups/huston/hustonlab/flp-chrimson_experiments/';
bodylblfile_list = '/groups/huston/hustonlab/flp-chrimson_experiments/fly2BodyAxis_lookupTable_Ben.csv';
crop_reg_file = '/groups/branson/bransonlab/mayank/stephen_copy/crop_regression_params.mat';
vid_sz = [512,768]; % ht wd
crop_sz = [[230,350];[350,350]];

dirs = {'fly_2146_to_2151_SS47486kir_multiaxis/fly21510', ...
  'fly_2197_to_2202_SS47486kir_multiaxis/fly22020', ...
'fly_2303_to_2311_emptysplitkir_pitchdown/fly23080',...
'fly_2317_to_2318_emptysplitkir_pitchdown/fly23171'};

%%
dirndx = 4;

bdir = fullfile(ndir,dirs{dirndx});
bdir2 = fullfile(gdir,dirs{dirndx});

[~,flydir] = fileparts(dirs{dirndx});
gg = textscan(flydir,'fly%d');
fly_num = gg{1};
tdir = fullfile('~/temp/sh_temp',flydir);

% calibration data
calib_list = '/groups/huston/hustonlab/flp-chrimson_experiments/fly2DLT_lookupTableStephen.csv';
ff = fopen(calib_list);
C = textscan(ff,'%d,%s');
fclose(ff);
fndx= find(C{1}==fly_num);
calib_file = C{2}{fndx};

% body label file
ff = fopen(bodylblfile_list);
B = textscan(ff,'%d,%s');
fndx = find(B{1}==fly_num);
bodylblfile = B{2}{fndx};
fclose(ff);

% crop locations
blbl = loadLbl(bodylblfile);
mm = SparseLabelArray.full(blbl.labeledpos{1});
neck_pts = zeros(2,2);
neck_pts(1,:) = mm(6,:,1);
neck_pts(2,:) = mm(16,:,1);

reg_params = load(crop_reg_file);
offsets = [0,20];
crops = zeros(2,4);
for vw = 1:2
  for cndx = 1:2    
    if cndx ==1
      cstr = 'x';
    else
      cstr = 'y';
    end
    cstr = sprintf('reg_view%d_%s',vw,cstr);
    cur_r = reg_params.(cstr);
    min_r = round(cur_r(1) + cur_r(2)*neck_pts(vw,cndx) - offsets(cndx));
    
    if min_r<1, min_r=1; end
    max_r = min_r + crop_sz(vw,cndx)-1;
    if max_r > vid_sz(3-cndx)
      max_r = vid_sz(3-cndx)-1;
      min_r = max_r - crop_sz(vw,cndx)+1;
    end
    crops(vw,cndx*2-1) = min_r;
    crops(vw,cndx*2) = max_r;
    
  end  
end
%crops =[75, 304, 48, 397;196, 545, 41, 390]; for 21510

vids = dir(sprintf('%s/C001*/*.avi',bdir));
if ~exist(tdir)
  mkdir(tdir);
end
%%

for ndx = 1:numel(vids)
  %%
  mov1 = fullfile(vids(ndx).folder,vids(ndx).name);
  mov2 = strrep(mov1,'C001','C002');
  tmov1 = fullfile(tdir,vids(ndx).name);
  tmov2 = strrep(tmov1,'C001','C002');
  
  out1 = strrep(tmov1,'.avi','.trk');
  out2 = strrep(tmov2,'.avi','.trk');
  copyfile(mov1,tmov1);
  copyfile(mov2,tmov2);
  toTrack = struct();
  toTrack.movfiles = {tmov1;tmov2}';
  toTrack.trkfiles = {out1;out2}';
  
  toTrack.calibrationfiles = {calib_file};
  toTrack.cropRois = {crops};
  
  trackStephen3D(tlbl_dir,'','toTrack',toTrack,'gpu_id',1,'save_raw',true);

  otr1 = strrep(mov1,ndir,gdir);
  otr1 = strrep(otr1,'.avi','.trk');
  otr2 = strrep(otr1,'C001','C002');
  
  tname = strrep(vids(ndx).name,'.avi','_old.trk');
  ntr1 = fullfile(tdir,tname);
  ntr2 = strrep(ntr1,'C001','C002');
  ot1 = load(otr1,'-mat');
  ot2 = load(otr2,'-mat');
  ot1_m = ot1;
  ot1_m.pTrk = {ot1.pTrk};
  ot1_m.pTrkTS = {ot1.pTrkTS};
  ot1_m.pTrkTag = {false(size(ot1.pTrkTag))};
  ot2_m = ot2;
  ot2_m.pTrk = {ot2.pTrk};
  ot2_m.pTrkTS = {ot2.pTrkTS};
  ot2_m.pTrkTag = {false(size(ot2.pTrkTag))};
  ot1_m.startframes = 1;
  ot1_m.endframes = size(ot1.pTrk,3);
  ot1_m.pTrkiTgt = 1;
  ot2_m.startframes = 1;
  ot2_m.endframes = size(ot1.pTrk,3);
  ot2_m.pTrkiTgt = 1;
  save(ntr1,'-struct','ot1_m');
  save(ntr2,'-struct','ot2_m');

end


%% compare trks

trks = dir(fullfile(bdir2,'*/C001*.trk'));
maxd = zeros(4,0);
for ndx = 1:numel(trks)
  %%
  otr1 = fullfile(trks(ndx).folder,trks(ndx).name);
  otr2 = strrep(otr1,'C001','C002');
  ntr1 = fullfile(tdir,trks(ndx).name);
  ntr2 = strrep(ntr1,'C001','C002');
  ot1 = load(otr1,'-mat');
  ot2 = load(otr2,'-mat');
  ot1_m = ot1;
  ot1_m.pTrk = {ot1.pTrk};
  ot1_m.pTrkTS = {ot1.pTrkTS};
  ot1_m.pTrkTag = {false(size(ot1.pTrkTag))};
  ot2_m = ot2;
  ot2_m.pTrk = {ot2.pTrk};
  ot2_m.pTrkTS = {ot2.pTrkTS};
  ot2_m.pTrkTag = {false(size(ot2.pTrkTag))};
  ot1_m.startframes = 1;
  ot1_m.endframes = size(ot1.pTrk,3);
  ot1_m.pTrkiTgt = 1;
  ot2_m.startframes = 1;
  ot2_m.endframes = size(ot1.pTrk,3);
  ot2_m.pTrkiTgt = 1;
  otr1_m = strrep(ntr1,'.trk','_old.trk');
  otr2_m = strrep(ntr2,'.trk','_old.trk');
%   save(otr1_m,'-struct','ot1_m');
%   save(otr2_m,'-struct','ot2_m');
  nt1 = load(ntr1,'-mat');
  nt2 = load(ntr2,'-mat');
  dd1 = squeeze(sum(sum(abs(ot1.pTrk-nt1.pTrk{1}),2),1));
  dd2 = squeeze(sum(sum(abs(ot2.pTrk-nt2.pTrk{1}),2),1));
  maxd(:,ndx) = [max(dd1),max(dd2),argmax(dd1),argmax(dd2)];
  
end

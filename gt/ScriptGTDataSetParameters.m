%ScriptGTDataSetParameters
cprdir = '/groups/branson/bransonlab/apt/experiments/res/cprgt20190407';

% info is nlabels x 1 x 3, where last 3 indices correspond to 
% movie, frm and trx_id

gtimdata = [];
assert(exist('exptype','var')>0);

nets = {'openpose','leap','deeplabcut','unet','resnet_unet','mdn','cpropt'};
legendnames = {'OpenPose','LEAP','DeepLabCut','U-net','Res-U-net','MDN','CPR'};
nnets = numel(nets);
colors = [
  0         0.4470    0.7410
  0.4660    0.6740    0.1880
  0.8500    0.3250    0.0980
  0.9290    0.6940    0.1250
  0.6350    0.0780    0.1840
  0.4940    0.1840    0.5560
  0.3010    0.7450    0.9330
  ];
prcs = [50,75,90,95,97];

idxnet = [1 2 3 7 4 5 6];
nets = nets(idxnet);
colors = colors(idxnet,:);
legendnames = legendnames(idxnet);
vwi = 1;
doAlignCoordSystem = false;
annoterrfile = '';
annoterrdata = [];
is3d = false;

switch exptype,
  case {'SHView0','SHView1'}
    if strcmp(exptype,'SHView0'),
      %gtfile_trainsize = '/nrs/branson/mayank/apt_cache/stephen_view0_trainsize.mat';
      gtfile_trainsize_cpr = fullfile(cprdir,'outputFINAL/stephen_view0_trainsize_withcpr.mat');
      gtfile_traintime_cpr = '';
      gtfile_trainsize = '/nrs/branson/mayank/apt_cache/stephen_view0_trainsize.mat';
      gtfile_traintime = '/nrs/branson/mayank/apt_cache/stephen_view0_time.mat';
      vwi = 1;
      annoterrfile = 'SHView0_AnnErrData20190718.mat';
    else
      %gtfile_trainsize = '/nrs/branson/mayank/apt_cache/stephen_view1_trainsize.mat';
      gtfile_trainsize_cpr = fullfile(cprdir,'outputFINAL/stephen_view1_trainsize_withcpr.mat');
      gtfile_traintime_cpr = '';
      gtfile_trainsize = '/nrs/branson/mayank/apt_cache/stephen_view1_trainsize.mat';
      gtfile_traintime = '/nrs/branson/mayank/apt_cache/stephen_view1_time.mat';
      vwi = 2;
      annoterrfile = 'SHView1_AnnErrData20190718.mat';
    end
    condinfofile = '/groups/branson/home/bransonk/tracking/code/APT/SHTrainGTInfo20190718.mat';
    gtdata_size = load(gtfile_trainsize);
    nlabels = size(gtdata_size.(nets{end}){end}.labels,1);
    npts = size(gtdata_size.(nets{end}){end}.labels,2);
    
    incondinfo = load(condinfofile);
    conddata = struct;
    % conditions:
    % enriched + activation
    % not enriched + activation
    % not activation
    % data types:
    % train
    % not train
    [conddata.data_cond,conddata.label_cond,datatypes,labeltypes] = SHGTInfo2CondData(incondinfo.gtinfo,true);
    pttypes = {'L. antenna tip',1
      'R. antenna tip',2
      'L. antenna base',3
      'R. antenna base',4
      'Proboscis roof',5};
    %     labeltypes = {'all',1};
    %     datatypes = {'all',1};
    maxerr = 60;
    lblfile = '/groups/branson/home/bransonk/tracking/code/APT/sh_trn4523_gtcomplete_cacheddata_bestPrms20180920_retrain20180920T123534_withGTres_mdn20190214_skeledges.lbl';
    freezeInfo = struct;
    freezeInfo.iMov = 502;
    freezeInfo.iTgt = 1;
    freezeInfo.frm = 746;
    doplotoverx = true;
    gtimdata = struct;
    gtimdata.ppdata = incondinfo.ppdata;
    gtimdata.tblPGT = incondinfo.tblPGT;
    
  case 'SH3D'    
    gtfile_trainsize_cpr = '';
    gtfile_traintime_cpr = '';
    gtfile_trainsize = '/groups/branson/home/bransonk/tracking/code/APT/stephen_3D_trainsize20190719.mat';
    gtfile_traintime = '/groups/branson/home/bransonk/tracking/code/APT/stephen_3D_traintime20190719.mat';
    vwi = 1;
    annoterrfile = '/groups/branson/home/bransonk/tracking/code/APT/SH3D_AnnErrData20190719.mat';

    condinfofile = '/groups/branson/home/bransonk/tracking/code/APT/SHTrainGTInfo20190718.mat';
    gtdata_size = load(gtfile_trainsize);
    nlabels = size(gtdata_size.(nets{end}){end}.labels,1);
    npts = size(gtdata_size.(nets{end}){end}.labels,2);
    
    incondinfo = load(condinfofile);
    conddata = struct;
    % conditions:
    % enriched + activation
    % not enriched + activation
    % not activation
    % data types:
    % train
    % not train
    [conddata.data_cond,conddata.label_cond,datatypes,labeltypes] = SHGTInfo2CondData(incondinfo.gtinfo,true);
    pttypes = {'L. antenna tip',1
      'R. antenna tip',2
      'L. antenna base',3
      'R. antenna base',4
      'Proboscis roof',5};
    %     labeltypes = {'all',1};
    %     datatypes = {'all',1};
    maxerr = [];
    lblfile = '/groups/branson/home/bransonk/tracking/code/APT/sh_trn4523_gtcomplete_cacheddata_bestPrms20180920_retrain20180920T123534_withGTres_mdn20190214_skeledges.lbl';
    freezeInfo = [];
    doplotoverx = true;
    gtimdata = struct;
    gtimdata.ppdata = incondinfo.ppdata;
    gtimdata.tblPGT = incondinfo.tblPGT;
    is3d = true;
    
  case 'FlyBubble'
    %gtfile_trainsize = '/nrs/branson/mayank/apt_cache/alice_view0_trainsize.mat';
    gtfile_trainsize_cpr = fullfile(cprdir,'outputFINAL/alice_view0_trainsize_withcpr.mat');
    gtfile_trainsize = '/nrs/branson/mayank/apt_cache/alice_view0_trainsize.mat';
    %gtfile_traintime = '/nrs/branson/mayank/apt_cache/alice_view0_time.mat';
    gtfile_traintime = '/nrs/branson/mayank/apt_cache/alice_view0_time.mat';
    gtfile_traintime_cpr = '';
    condinfofile = '/nrs/branson/mayank/apt_cache/multitarget_bubble/multitarget_bubble_expandedbehavior_20180425_condinfo.mat';
    gtimagefile = '/groups/branson/home/bransonk/tracking/code/APT/FlyBubbleGTData20190524.mat';
    gtimdata = load(gtimagefile);
    conddata = load(condinfofile);
    
    pttypes = {'head',[1,2,3]
      'body',[4,5,6,7]
      'middle leg joint 1',[8,10]
      'middle leg joint 2',[9,11]
      'front leg tarsi',[12,17]
      'middle leg tarsi',[13,16]
      'back leg tarsi',[14,15]};
    labeltypes = {'moving',1
      'grooming',2
      'close',3
      'all',[1,2,3]};
    datatypes = {'same fly',1
      'same genotype',2
      'different genotype',3
      'all',[1,2,3]};
    lblfile = '/groups/branson/home/bransonk/tracking/code/APT/multitarget_bubble_expandedbehavior_20180425_FxdErrs_OptoParams20181126_mdn20190214_skeledges.lbl';
    maxerr = 30;
    doplotoverx = true;
    gtdata_size = load(gtfile_trainsize);
    npts = size(gtdata_size.(nets{end}){end}.labels,2);
    annoterrfile = 'AnnotErrData20190614.mat';
    
  case {'RFView0','RFView1'}
    if strcmp(exptype,'RFView0'),
      %gtfile_trainsize = '/nrs/branson/mayank/apt_cache/stephen_view0_trainsize.mat';
      gtfile_trainsize_cpr = '/groups/branson/bransonlab/apt/experiments/res/cpr_xv_20190504/romn/out/xv_romnproj_alcpreasyprms_tblcvi_romn_split_romn_20190515T173224.mat';%fullfile(cprdir,'outputFINAL/stephen_view0_trainsize_withcpr.mat');
      gtfile_traintime_cpr = '';
      gtfile_trainsize = '/nrs/branson/mayank/apt_cache/romain_view0_cv.mat';
      gtfile_traintime = '';
      vwi = 1;
    else
      %gtfile_trainsize = '/nrs/branson/mayank/apt_cache/stephen_view1_trainsize.mat';
      gtfile_trainsize_cpr = '/groups/branson/bransonlab/apt/experiments/res/cpr_xv_20190504/romn/out/xv_romnproj_alcpreasyprms_tblcvi_romn_split_romn_20190515T173224.mat';%fullfile(cprdir,'outputFINAL/stephen_view0_trainsize_withcpr.mat');
      gtfile_trainsize = '/nrs/branson/mayank/apt_cache/romain_view1_cv.mat';
      gtfile_traintime = '';
      vwi = 2;
    end
    condinfofile = '';
    gtdata_size = load(gtfile_trainsize);
    nlabels = size(gtdata_size.(nets{end}){end}.labels,1);
    npts = size(gtdata_size.(nets{end}){end}.labels,2);
    
    conddata = [];
    labeltypes = {};
    datatypes = {};
    
    
    %     pttypes = {'L. antenna tip',1
    %       'R. antenna tip',2
    %       'L. antenna base',3
    %       'R. antenna base',4
    %       'Proboscis roof',5};
    %     labeltypes = {'all',1};
    %     datatypes = {'all',1};
    maxerr = [];
    lblfile = '/groups/branson/bransonlab/apt/experiments/data/romainTrackNov18_updateDec06_al_portable_mdn60k_openposewking_newmacro.lbl';
    %lblfile = '/groups/branson/bransonlab/apt/experiments/res/romain_viewpref_3dpostproc_20190522/romainTrackNov18_al_portable_mp4s_withExpTriResMovs134_20190522.lbl';
    gtimagefile = '/groups/branson/home/bransonk/tracking/code/APT/RomainTrainCVInfo20190419.mat';
    freezeInfo = struct;
    freezeInfo.iMov = 1;
    freezeInfo.iTgt = 1;
    freezeInfo.frm = 302;
    freezeInfo.clim = [0,192];
    doplotoverx = false;
    gtimdata = load(gtimagefile);
    
    pttypes = {'abdomen',19
      'front leg joint 1',[13,16]
      'front leg joint 2',[7,10]
      'front leg tarsi',[1,4]
      'middle leg joint 1',[14,17]
      'middle leg joint 2',[8,11]
      'middle leg tarsi',[2,5]
      'back leg joint 1',[15,18]
      'back leg joint 2',[9,12]
      'back leg tarsi',[3,6]};
    
    
  case 'RF3D'
    gtfile_trainsize_cpr = '';
    gtfile_traintime_cpr = '';
    gtfile_trainsize = '/groups/branson/home/bransonk/tracking/code/APT/RF3D_trainsize20190722.mat';
    gtfile_traintime = '';
    vwi = 1;
    annoterrfile = '';
    
    condinfofile = '';
    gtdata_size = load(gtfile_trainsize);
    nlabels = size(gtdata_size.(nets{end}){end}.labels,1);
    npts = size(gtdata_size.(nets{end}){end}.labels,2);

    conddata = [];
    maxerr = [];
    lblfile = '/groups/branson/bransonlab/apt/experiments/data/romainTrackNov18_updateDec06_al_portable_mdn60k_openposewking_newmacro.lbl';
    gtimagefile = '/groups/branson/home/bransonk/tracking/code/APT/RomainTrainCVInfo20190419.mat';
    freezeInfo = [];
    is3d = true;
    doplotoverx = false;
    gtimdata = load(gtimagefile);
    
    pttypes = {'abdomen',19
      'front leg joint 1',[13,16]
      'front leg joint 2',[7,10]
      'front leg tarsi',[1,4]
      'middle leg joint 1',[14,17]
      'middle leg joint 2',[8,11]
      'middle leg tarsi',[2,5]
      'back leg joint 1',[15,18]
      'back leg joint 2',[9,12]
      'back leg tarsi',[3,6]};
    
  case 'Larva',
    
    gtfile_trainsize_cpr = '/groups/branson/bransonlab/apt/experiments/res/cpr_xv_20190504/larv/out/xv_Larva94A04_CM_tbltrn_larv_split_larv_prm_larv_ar_20190515T093243.mat';
    gtfile_trainsize = '/nrs/branson/mayank/apt_cache/larva_view0_cv.mat';
    gtfile_traintime = '';
    gtimagefile = '/groups/branson/home/bransonk/tracking/code/APT/LarvaTrainCVInfo20190419.mat';
    gtdata_size = load(gtfile_trainsize);
    nlabels = size(gtdata_size.(nets{end}){end}.labels,1);
    
    conddata = [];
    % conditions:
    % enriched + activation
    % not enriched + activation
    % not activation
    % data types:
    % train
    % not train
    labeltypes = {};
    datatypes = {};
    pttypes = {'outside',[1:2:13,16:2:28]
      'inside',[2:2:14,15:2:27]};
    %     pttypes = {'L. antenna tip',1
    %       'R. antenna tip',2
    %       'L. antenna base',3
    %       'R. antenna base',4
    %       'Proboscis roof',5};
    %     labeltypes = {'all',1};
    %     datatypes = {'all',1};
    maxerr = [];
    gtimdata = load(gtimagefile);
    
    lblfile = '/groups/branson/bransonlab/larvalmuscle_2018/APT_Projects/Larva94A04_CM_fixedmovies.lbl';
    freezeInfo = struct;
    freezeInfo.iMov = 4;
    freezeInfo.iTgt = 1;
    freezeInfo.frm = 5;
    freezeInfo.axes_curr.XLim = [745,1584];
    freezeInfo.axes_curr.YLim = [514,1353];
    doplotoverx = false;
    
  case 'Roian'
    %gtfile_trainsize = '/nrs/branson/mayank/apt_cache/stephen_view0_trainsize.mat';
    gtfile_trainsize_cpr = '/groups/branson/bransonlab/apt/experiments/res/cpr_xv_20190504/sere/out/xv_sere_al_cprparamsbigger_20190514_tblcvi_sere_split_sere_20190515T094434.mat';
    gtfile_traintime_cpr = '';
    gtfile_trainsize = '/nrs/branson/mayank/apt_cache/roian_view0_cv.mat';
    gtfile_traintime = '';
    vwi = 1;
    condinfofile = '';
    gtdata_size = load(gtfile_trainsize);
    nlabels = size(gtdata_size.(nets{end}){end}.labels,1);
    npts = size(gtdata_size.(nets{end}){end}.labels,2);
    
    conddata = [];
    % conditions:
    % enriched + activation
    % not enriched + activation
    % not activation
    % data types:
    % train
    % not train
    labeltypes = {};
    datatypes = {};
    %     pttypes = {'L. antenna tip',1
    %       'R. antenna tip',2
    %       'L. antenna base',3
    %       'R. antenna base',4
    %       'Proboscis roof',5};
    %     labeltypes = {'all',1};
    %     datatypes = {'all',1};
    maxerr = [];
    lblfile = '/groups/branson/bransonlab/apt/experiments/data/roian_apt.lbl';
    gtimagefile = '/groups/branson/home/bransonk/tracking/code/APT/RoianTrainCVInfo20190420.mat';
    freezeInfo = struct;
    freezeInfo.iMov = 1;
    freezeInfo.iTgt = 1;
    freezeInfo.frm = 1101;
    doplotoverx = false;
    gtimdata = load(gtimagefile);
    
    pttypes = {'nose',1
      'tail',2
      'ear',[3,4]};
    
  case {'BSView0x','BSView1x','BSView2x'}
    if strcmp(exptype,'BSView0x'),
      gtfile_trainsize_cpr = '/groups/branson/bransonlab/apt/experiments/res/cpr_xv_20190504/brit/out/xv_wheel_rig_tracker_DEEP_cam0_tbltrn_brit_vw1_split_brit_vw1_prm_brit_al_20190515T184617.mat';
      gtfile_traintime_cpr = '';
      gtfile_trainsize = '/nrs/branson/mayank/apt_cache/brit0_view0_cv.mat';
      gtfile_traintime = '';
      vwi = 1;
      lblfile = '/groups/branson/bransonlab/apt/experiments/data/wheel_rig_tracker_DEEP_cam0.lbl';
      pttypes = {'Front foot',[1,2]
        'Back foot',[3,4]
        'Tail',5};
    elseif strcmp(exptype,'BSView1x'),
      gtfile_trainsize_cpr = '/groups/branson/bransonlab/apt/experiments/res/cpr_xv_20190504/brit/out/xv_wheel_rig_tracker_DEEP_cam1_tbltrn_brit_vw2_split_brit_vw2_prm_brit_al_20190515T184622.mat';
      gtfile_traintime_cpr = '';
      gtfile_trainsize = '/nrs/branson/mayank/apt_cache/brit1_view0_cv.mat';
      gtfile_traintime = '';
      vwi = 1;
      lblfile = '/groups/branson/bransonlab/apt/experiments/data/wheel_rig_tracker_DEEP_cam1.lbl';
      pttypes = {'Front foot',[1,2]};
    else
      gtfile_trainsize_cpr = '/groups/branson/bransonlab/apt/experiments/res/cpr_xv_20190504/brit/out/xv_wheel_rig_tracker_DEEP_cam2_tbltrn_brit_vw3_split_brit_vw3_prm_brit_al_20190515T184819.mat';
      gtfile_traintime_cpr = '';
      gtfile_trainsize = '/nrs/branson/mayank/apt_cache/brit2_view0_cv.mat';
      gtfile_traintime = '';
      vwi = 1;
      lblfile = '/groups/branson/bransonlab/apt/experiments/data/wheel_rig_tracker_DEEP_cam2.lbl';
      pttypes = {'Back foot',[1,2]
        'Tail',3};
    end
    condinfofile = '';
    gtdata_size = load(gtfile_trainsize);
    nlabels = size(gtdata_size.(nets{end}){end}.labels,1);
    npts = size(gtdata_size.(nets{end}){end}.labels,2);
    
    conddata = [];
    % conditions:
    % enriched + activation
    % not enriched + activation
    % not activation
    % data types:
    % train
    % not train
    labeltypes = {};
    datatypes = {};
    %     pttypes = {'L. antenna tip',1
    %       'R. antenna tip',2
    %       'L. antenna base',3
    %       'R. antenna base',4
    %       'Proboscis roof',5};
    %     labeltypes = {'all',1};
    %     datatypes = {'all',1};
    maxerr = [];
    gtimagefile = '/groups/branson/home/bransonk/tracking/code/APT/BSTrainCVInfo20190416.mat';
    
    freezeInfo = struct;
    freezeInfo.i = 1;
    doplotoverx = false;
    gtimdatain = load(gtimagefile);
    realvwi = str2double(regexp(exptype,'View(\d+)x','once','tokens'))+1;
    gtimdata = struct;
    gtimdata.cvi = gtimdatain.cvidx{realvwi};
    gtimdata.ppdata = gtimdatain.ppdatas{realvwi};
    gtimdata.tblPGT = gtimdatain.tblPGTs{realvwi};
    gtimdata.frame = gtimdata.tblPGT.frm;
    gtimdata.movieidx = gtimdata.tblPGT.mov;
    gtimdata.movies = gtimdatain.trnmovies{realvwi};
    gtimdata.target = gtimdata.tblPGT.iTgt;
    
    
    %     pttypes = {'abdomen',19
    %       'front leg joint 1',[13,16]
    %       'front leg joint 2',[7,10]
    %       'front leg tarsi',[1,4]
    %       'middle leg joint 1',[14,17]
    %       'middle leg joint 2',[8,11]
    %       'middle leg tarsi',[2,5]
    %       'back leg joint 1',[15,18]
    %       'back leg joint 2',[9,12]
    %       'back leg tarsi',[3,6]};
    
  case 'FlyBubbleMDNvsDLC',
    gtfile_trainsize = '/groups/branson/home/robiea/Projects_data/Labeler_APT/Austin_labelerprojects_expandedbehaviors/GT/MDNvsDLC_20190530.mat';
    gtfile_trainsize_cpr = '';
    gtfile_traintime = '';
    gtfile_traintime_cpr = '';
    conddata = [];
    gtimagefile = '/groups/branson/home/bransonk/tracking/code/APT/FlyBubbleMDNvsDLC_gtimdata_20190531.mat';
    gtimdata = load(gtimagefile);
    
    nets = {'DLC','MDN'};
    legendnames = {'DeepLabCut','MDN'};
    nnets = numel(nets);
    colors = [
      0.8500    0.3250    0.0980
      0.4940    0.1840    0.5560
      ];
    labeltypes = {};
    datatypes = {};
    pttypes = {'head',[1,2,3]
      'body',[4,5,6,7]
      'middle leg joint 1',[8,10]
      'middle leg joint 2',[9,11]
      'front leg tarsi',[12,17]
      'middle leg tarsi',[13,16]
      'back leg tarsi',[14,15]};
    lblfile = '/groups/branson/home/bransonk/tracking/code/APT/multitarget_bubble_expandedbehavior_20180425_FxdErrs_OptoParams20181126_mdn20190214_skeledges.lbl';
    maxerr = [];
    doplotoverx = false;
    doAlignCoordSystem = true;
    
    
  otherwise
    error('Unknown exp type %s',exptype);
    
end


% nets = {'openpose','deeplabcut','unet','mdn'};
% nnets = numel(nets);
% colors = [
%   0    0.4470    0.7410
%   0.8500    0.3250    0.0980
%   0.9290    0.6940    0.1250
%   0.4940    0.1840    0.5560
%   ];
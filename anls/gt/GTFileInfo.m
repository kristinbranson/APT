function gtfileinfo = GTFileInfo(exptype)

gtfile_final0 = '';
cprdir = '/groups/branson/bransonlab/apt/experiments/res/cprgt20190407';
gtimagefile = '';
annoterrfile = '';

switch exptype,
  case {'SHView0','SHView1'}
    if strcmp(exptype,'SHView0'),
      vwi = 1;
    else
      vwi = 2;
    end
    
    gtfile_trainsize_cpr = fullfile(cprdir,sprintf('outputFINAL/stephen_view%d_trainsize_withcpr.mat',vwi-1));
    gtfile_cpr = fullfile(cprdir,sprintf('outputFINAL/stephen_view%d_trainsize_withcpr.mat',vwi-1));
    gtfile_traintime_cpr = '';
    gtfile_trainsize = sprintf('/groups/branson/bransonlab/mayank/apt_results/stephen_deepnet_20200717_view%d_trainsize20200722.mat',vwi-1);
    gtfile_traintime = '';
    %gtfile_trainsize = '/nrs/branson/mayank/apt_cache/stephen_view0_trainsize.mat';
    %gtfile_traintime = '/nrs/branson/mayank/apt_cache/stephen_view0_time.mat';
      
    gtfile_final0 = sprintf('/groups/branson/bransonlab/mayank/apt_results/stephen_deepnet_20200706_view%d_time20200708.mat',vwi-1);
    
    gtfile_final = sprintf('stephen_deepnet_20200706_view%d_time20200708_and_20200722_final.mat',vwi-1);
    annoterrfile = fullfile('/groups/branson/home/bransonk/tracking/code/APT',sprintf('SHView%d_AnnErrData20190718.mat',vwi-1));
    condinfofile = '/groups/branson/home/bransonk/tracking/code/APT/SHTrainGTInfo20190718.mat';
    
    %lblfile = '/groups/branson/home/bransonk/tracking/code/APT/sh_trn4523_gtcomplete_cacheddata_bestPrms20180920_retrain20180920T123534_withGTres_mdn20190214_skeledges.lbl';
    lblfile = '/groups/branson/bransonlab/apt/experiments/data/sh_trn4992_gtcomplete_cacheddata_updated20200807.lbl';
    
  case 'SH3D'    
    gtfile_trainsize_cpr = '';
    gtfile_cpr = '';
    gtfile_traintime_cpr = '';
%     gtfile_trainsize = '/groups/branson/home/bransonk/tracking/code/APT/stephen_3D_trainsize20190719.mat';
%     gtfile_traintime = '/groups/branson/home/bransonk/tracking/code/APT/stephen_3D_traintime20190719.mat';
    gtfile_trainsize = '/groups/branson/home/bransonk/tracking/code/APT/stephen_3D_trainsize20200810.mat';
    gtfile_traintime = '';
    %gtfile_traintime = '/groups/branson/home/bransonk/tracking/code/APT/stephen_3D_traintime20200810.mat';
    gtfile_final = '/groups/branson/home/bransonk/tracking/code/APT/stephen_3D_final20200810.mat';
    annoterrfile = '/groups/branson/home/bransonk/tracking/code/APT/SH3D_AnnErrData20200810.mat';
    condinfofile = '/groups/branson/home/bransonk/tracking/code/APT/SHTrainGTInfo20190718.mat';
    %lblfile = '/groups/branson/home/bransonk/tracking/code/APT/sh_trn4523_gtcomplete_cacheddata_bestPrms20180920_retrain20180920T123534_withGTres_mdn20190214_skeledges.lbl';
    lblfile = '/groups/branson/bransonlab/apt/experiments/data/sh_trn4992_gtcomplete_cacheddata_updated20200807.lbl';
  case 'FlyBubble'
    %gtfile_trainsize = '/nrs/branson/mayank/apt_cache/alice_view0_trainsize.mat';
    gtfile_trainsize_cpr = fullfile(cprdir,'outputFINAL/alice_view0_trainsize_withcpr.mat');
    gtfile_cpr = fullfile(cprdir,'outputFINAL/alice_view0_trainsize_withcpr.mat');
    %gtfile_trainsize = '/nrs/branson/mayank/apt_cache/alice_view0_trainsize.mat';
    gtfile_trainsize = '/groups/branson/bransonlab/mayank/apt_results/alice_deepnet_20200716_view0_trainsize20200717.mat';
    %gtfile_traintime = '/nrs/branson/mayank/apt_cache/alice_view0_time.mat';
    %gtfile_traintime = '/nrs/branson/mayank/apt_cache/alice_view0_time.mat';
    gtfile_traintime = '';
    gtfile_final0 = '/groups/branson/bransonlab/mayank/apt_results/alice_deepnet_20200706_view0_time20200708.mat';
    gtfile_final = 'alice_deepnet_20200716_view0_20200708_and_20200717_final.mat';
    gtfile_traintime_cpr = '';
    condinfofile = '/nrs/branson/mayank/apt_cache/multitarget_bubble/multitarget_bubble_expandedbehavior_20180425_condinfo.mat';
    gtimagefile = '/groups/branson/home/bransonk/tracking/code/APT/FlyBubbleGTData20190524.mat';
    %lblfile = '/groups/branson/home/bransonk/tracking/code/APT/multitarget_bubble_expandedbehavior_20180425_FxdErrs_OptoParams20181126_mdn20190214_skeledges.lbl';
    lblfile = '/groups/branson/bransonlab/apt/experiments/data/multitarget_bubble_expandedbehavior_20180425_FxdErrs_OptoParams20200317.lbl';
    annoterrfile = 'AnnotErrData20190614.mat';
    
  case {'RFView0','RFView1'}
    if strcmp(exptype,'RFView0'),
      %gtfile_trainsize = '/nrs/branson/mayank/apt_cache/stephen_view0_trainsize.mat';
      gtfile_trainsize_cpr = '/groups/branson/bransonlab/apt/experiments/res/cpr_xv_20190504/romn/out/xv_romnproj_alcpreasyprms_tblcvi_romn_split_romn_20190515T173224.mat';%fullfile(cprdir,'outputFINAL/stephen_view0_trainsize_withcpr.mat');
      gtfile_traintime_cpr = '';
      gtfile_cpr = '/groups/branson/bransonlab/apt/experiments/res/cpr_xv_20190504/romn/out/xv_romnproj_alcpreasyprms_tblcvi_romn_split_romn_20190515T173224.mat';%fullfile(cprdir,'outputFINAL/stephen_view0_trainsize_withcpr.mat');
      %gtfile_trainsize = '/nrs/branson/mayank/apt_cache/romain_view0_cv.mat';
      gtfile_trainsize = '';
      %gtfile_final = '/nrs/branson/mayank/apt_cache/romain_view0_cv.mat';
      gtfile_final = 'TODO';
      gtfile_traintime = '';
    else
      %gtfile_trainsize = '/nrs/branson/mayank/apt_cache/stephen_view1_trainsize.mat';
      gtfile_trainsize_cpr = '/groups/branson/bransonlab/apt/experiments/res/cpr_xv_20190504/romn/out/xv_romnproj_alcpreasyprms_tblcvi_romn_split_romn_20190515T173224.mat';%fullfile(cprdir,'outputFINAL/stephen_view0_trainsize_withcpr.mat');
      gtfile_trainsize = '';
      gtfile_cpr = '/groups/branson/bransonlab/apt/experiments/res/cpr_xv_20190504/romn/out/xv_romnproj_alcpreasyprms_tblcvi_romn_split_romn_20190515T173224.mat';%fullfile(cprdir,'outputFINAL/stephen_view0_trainsize_withcpr.mat');      %gtfile_final = '/nrs/branson/mayank/apt_cache/romain_view1_cv.mat';
      gtfile_final = 'TODO';
      gtfile_traintime = '';
    end
    condinfofile = '';
    lblfile = '/groups/branson/bransonlab/apt/experiments/data/romainTrackNov18_updateDec06_al_portable_mdn60k_openposewking_newmacro.lbl';
    %lblfile = '/groups/branson/bransonlab/apt/experiments/res/romain_viewpref_3dpostproc_20190522/romainTrackNov18_al_portable_mp4s_withExpTriResMovs134_20190522.lbl';
    gtimagefile = '/groups/branson/home/bransonk/tracking/code/APT/RomainTrainCVInfo20190419.mat';    
  case 'RF3D'
    gtfile_trainsize_cpr = '';
    gtfile_traintime_cpr = '';
    gtfile_trainsize = '';
    %gtfile_final = '/groups/branson/home/bransonk/tracking/code/APT/RF3D_trainsize20190722.mat';
    gtfile_final = '/groups/branson/home/bransonk/tracking/code/APT/RF3D_trainsize20200810.mat';
    gtfile_traintime = '';
    annoterrfile = '';    
    condinfofile = '';
    lblfile = '/groups/branson/bransonlab/apt/experiments/data/romainTrackNov18_updateDec06_al_portable_mdn60k_openposewking_newmacro.lbl';
    gtimagefile = '/groups/branson/home/bransonk/tracking/code/APT/RomainTrainCVInfo20190419.mat';
  case 'Larva',
    
    gtfile_trainsize_cpr = '/groups/branson/bransonlab/apt/experiments/res/cpr_xv_20190504/larv/out/xv_Larva94A04_CM_tbltrn_larv_split_larv_prm_larv_ar_20190515T093243.mat';
    gtfile_cpr = '/groups/branson/bransonlab/apt/experiments/res/cpr_xv_20190504/larv/out/xv_Larva94A04_CM_tbltrn_larv_split_larv_prm_larv_ar_20190515T093243.mat';
    gtfile_final = 'TODO';
    gtfile_trainsize = '';
    gtfile_traintime = '';
    gtimagefile = '/groups/branson/home/bransonk/tracking/code/APT/LarvaTrainCVInfo20190419.mat';
    lblfile = '/groups/branson/bransonlab/larvalmuscle_2018/APT_Projects/Larva94A04_CM_fixedmovies.lbl';
    
  case 'Roian'
    %gtfile_trainsize = '/nrs/branson/mayank/apt_cache/stephen_view0_trainsize.mat';
    %gtfile_trainsize_cpr = '/groups/branson/bransonlab/apt/experiments/res/cpr_xv_20190504/sere/out/xv_sere_al_cprparamsbigger_20190514_tblcvi_sere_split_sere_20190515T094434.mat';
    gtfile_trainsize_cpr = '';
    gtfile_traintime_cpr = '';
    gtfile_cpr = '';
    %gtfile_cpr = '/groups/branson/bransonlab/apt/experiments/res/cpr_xv_20190504/sere/out/xv_sere_al_cprparamsbigger_20190514_tblcvi_sere_split_sere_20190515T094434.mat';
    gtfile_trainsize = '';
    %gtfile_final = '/nrs/branson/mayank/apt_cache/roian_view0_cv.mat';
    gtfile_final = '/groups/branson/bransonlab/mayank/apt_results/roian_deepnet_20200712_view0_cv20200716.mat';
    gtfile_traintime = '';
    condinfofile = '';
    %lblfile = '/groups/branson/bransonlab/apt/experiments/data/roian_apt.lbl';
    lblfile = '/groups/branson/bransonlab/apt/experiments/data/four_points_all_mouse_linux_tracker_updated20200423.lbl';
    gtimagefile = '/groups/branson/home/bransonk/tracking/code/APT/RoianTrainCVInfo20190420.mat';
  case {'BSView0x','BSView1x','BSView2x'}
    if strcmp(exptype,'BSView0x'),
      gtfile_trainsize_cpr = '/groups/branson/bransonlab/apt/experiments/res/cpr_xv_20190504/brit/out/xv_wheel_rig_tracker_DEEP_cam0_tbltrn_brit_vw1_split_brit_vw1_prm_brit_al_20190515T184617.mat';
      gtfile_traintime_cpr = '';
      gtfile_cpr = '/groups/branson/bransonlab/apt/experiments/res/cpr_xv_20190504/brit/out/xv_wheel_rig_tracker_DEEP_cam0_tbltrn_brit_vw1_split_brit_vw1_prm_brit_al_20190515T184617.mat';
      gtfile_trainsize = '';
      gtfile_traintime = '';
      %gtfile_final = '/nrs/branson/mayank/apt_cache/brit0_view0_cv.mat';
      gtfile_final = '/groups/branson/bransonlab/mayank/apt_results/brit0_deepnet_20200710_view0_cv20200717.mat';
      %lblfile = '/groups/branson/bransonlab/apt/experiments/data/wheel_rig_tracker_DEEP_cam0.lbl';
      lblfile = '/groups/branson/bransonlab/apt/experiments/data/wheel_rig_tracker_DEEP_cam0_20200318_compress20200327.lbl';
    elseif strcmp(exptype,'BSView1x'),
      gtfile_trainsize_cpr = '/groups/branson/bransonlab/apt/experiments/res/cpr_xv_20190504/brit/out/xv_wheel_rig_tracker_DEEP_cam1_tbltrn_brit_vw2_split_brit_vw2_prm_brit_al_20190515T184622.mat';
      gtfile_cpr = '/groups/branson/bransonlab/apt/experiments/res/cpr_xv_20190504/brit/out/xv_wheel_rig_tracker_DEEP_cam1_tbltrn_brit_vw2_split_brit_vw2_prm_brit_al_20190515T184622.mat';
      gtfile_traintime_cpr = '';
      gtfile_trainsize = '';
      gtfile_traintime = '';
      %gtfile_final = '/nrs/branson/mayank/apt_cache/brit1_view0_cv.mat';
      gtfile_final = '/groups/branson/bransonlab/mayank/apt_results/brit1_deepnet_20200710_view0_cv20200717.mat';
      %lblfile = '/groups/branson/bransonlab/apt/experiments/data/wheel_rig_tracker_DEEP_cam1.lbl';
      lblfile = '/groups/branson/bransonlab/apt/experiments/data/wheel_rig_tracker_DEEP_cam1_20209327_compress20200330.lbl';
    else
      gtfile_trainsize_cpr = '/groups/branson/bransonlab/apt/experiments/res/cpr_xv_20190504/brit/out/xv_wheel_rig_tracker_DEEP_cam2_tbltrn_brit_vw3_split_brit_vw3_prm_brit_al_20190515T184819.mat';
      gtfile_traintime_cpr = '';
      gtfile_cpr = '/groups/branson/bransonlab/apt/experiments/res/cpr_xv_20190504/brit/out/xv_wheel_rig_tracker_DEEP_cam2_tbltrn_brit_vw3_split_brit_vw3_prm_brit_al_20190515T184819.mat';
      gtfile_trainsize = '';
      gtfile_traintime = '';
      %gtfile_final = '/nrs/branson/mayank/apt_cache/brit2_view0_cv.mat';
      gtfile_final = '/groups/branson/bransonlab/mayank/apt_results/brit2_deepnet_20200710_view0_cv20200717.mat';
      %lblfile = '/groups/branson/bransonlab/apt/experiments/data/wheel_rig_tracker_DEEP_cam2.lbl';
      lblfile = '/groups/branson/bransonlab/apt/experiments/data/wheel_rig_tracker_DEEP_cam2_20209327_compress20200330.lbl';
    end
    condinfofile = '';
    gtimagefile = '/groups/branson/home/bransonk/tracking/code/APT/BSTrainCVInfo20190416.mat';
  case 'FlyBubbleMDNvsDLC',
    %gtfile_trainsize = '/groups/branson/home/robiea/Projects_data/Labeler_APT/Austin_labelerprojects_expandedbehaviors/GT/MDNvsDLC_20190530.mat';
    gtfile_trainsize = '/groups/branson/bransonlab/mayank/apt_results/alice_difficult_deepnet_20200706_view0_time_diff20200708.mat';
    gtfile_trainsize_cpr = '';
    gtfile_cpr = '';
    gtfile_traintime = '';
    gtfile_traintime_cpr = '';
    gtimagefile = '/groups/branson/home/bransonk/tracking/code/APT/FlyBubbleMDNvsDLC_gtimdata_20190531.mat';
    lblfile = '/groups/branson/home/bransonk/tracking/code/APT/multitarget_bubble_expandedbehavior_20180425_FxdErrs_OptoParams20181126_mdn20190214_skeledges.lbl';
  otherwise
    error('Unknown exp type %s',exptype);
   
end

gtfileinfo = struct;
gtfileinfo.trainsize_cpr = gtfile_trainsize_cpr;
gtfileinfo.cpr = gtfile_cpr;
gtfileinfo.traintime_cpr = gtfile_traintime_cpr;
gtfileinfo.trainsize = gtfile_trainsize;
gtfileinfo.traintime = gtfile_traintime;
gtfileinfo.final0 = gtfile_final0;
gtfileinfo.final = gtfile_final;
gtfileinfo.annoterrfile = annoterrfile;
gtfileinfo.condinfofile = condinfofile;
gtfileinfo.lblfile = lblfile;
gtfileinfo.gtimagefile = gtimagefile;

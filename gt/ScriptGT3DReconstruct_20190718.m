exptype = 'RF';
cprdir = '/groups/branson/bransonlab/apt/experiments/res/cprgt20190407';
isshexp = false;
switch exptype,
  case 'SH'
    gtfile_trainsize_cpr = {fullfile(cprdir,'outputFINAL/stephen_view0_trainsize_withcpr.mat'),fullfile(cprdir,'outputFINAL/stephen_view1_trainsize_withcpr.mat')};
    gtfile_trainsize = {'/nrs/branson/mayank/apt_cache/stephen_view0_trainsize.mat','/nrs/branson/mayank/apt_cache/stephen_view1_trainsize.mat'};
    gtfile_traintime = {'/nrs/branson/mayank/apt_cache/stephen_view0_time.mat','/nrs/branson/mayank/apt_cache/stephen_view1_time.mat'};
    annoterrfile = {'SHView0_AnnErrData20190718.mat','SHView1_AnnErrData20190718.mat'};
    condinfofile = '/groups/branson/home/bransonk/tracking/code/APT/SHTrainGTInfo20190718.mat';
    lblfile = '/groups/branson/home/bransonk/tracking/code/APT/sh_trn4523_gtcomplete_cacheddata_bestPrms20180920_retrain20180920T123534_withGTres_mdn20190214_skeledges.lbl';
    isshexp = true;
    incondinfo = load(condinfofile);
    gtinfo = incondinfo.gtinfo;
    
  case 'SHsingle'
    
    gtfile_trainsize = {'/nrs/branson/mayank/apt_cache/stephen_view0_single.mat',...
      '/nrs/branson/mayank/apt_cache/stephen_view1_single.mat'};
    gtfile_traintime = {};
    gtfile_trainsize_cpr = {};
    gtfile_traintime_cpr = {};
    annoterrfile = {'SHView0_AnnErrData20190718.mat','SHView1_AnnErrData20190718.mat'};
    condinfofile = '/groups/branson/home/bransonk/tracking/code/APT/SHTrainGTInfo20190718.mat';
    lblfile = '/groups/branson/home/bransonk/tracking/code/APT/sh_trn4523_gtcomplete_cacheddata_bestPrms20180920_retrain20180920T123534_withGTres_mdn20190214_skeledges.lbl';
    isshexp = true;
    incondinfo = load(condinfofile);
    gtinfo = incondinfo.gtinfo;

  case 'RF'
    
    gtfile_trainsize_cpr = {'/groups/branson/bransonlab/apt/experiments/res/cpr_xv_20190504/romn/out/xv_romnproj_alcpreasyprms_tblcvi_romn_split_romn_20190515T173224.mat',...
      '/groups/branson/bransonlab/apt/experiments/res/cpr_xv_20190504/romn/out/xv_romnproj_alcpreasyprms_tblcvi_romn_split_romn_20190515T173224.mat'}; % same file twice
    gtfile_traintime_cpr = '';
    gtfile_trainsize = {'/nrs/branson/mayank/apt_cache/romain_view0_cv.mat','/nrs/branson/mayank/apt_cache/romain_view1_cv.mat'};
    gtfile_traintime = {};
    annoterrfile = {};
    trninfofile = '/groups/branson/home/bransonk/tracking/code/APT/RomainTrainCVInfo20190419.mat';
    gtinfo = load(trninfofile);
    conddata = [];
    labeltypes = {};
    datatypes = {};
    lblfile = '/groups/branson/bransonlab/apt/experiments/data/romainTrackNov18_updateDec06_al_portable_mdn60k_openposewking_newmacro.lbl';
    %lblfile = '/groups/branson/bransonlab/apt/experiments/res/romain_viewpref_3dpostproc_20190522/romainTrackNov18_al_portable_mp4s_withExpTriResMovs134_20190522.lbl';

  otherwise
    error('Unknown exptype %s',exptype);
end
nviews = numel(gtfile_trainsize);

try
  lObj = load(lblfile,'-mat');
catch
  tmploc = tempname;
  mkdir(tmploc);
  untar(lblfile,tmploc);
  lObj = load(fullfile(tmploc,'label_file.lbl'),'-mat');
  unix(sprintf('rm -r %s',tmploc));
end
ptcolors = lObj.cfg.LabelPointsPlot.Colors;
lObj.labeledpos = cellfun(@SparseLabelArray.full,lObj.labeledpos,'uni',0);

gtdata2d_size = cell(1,nviews);
for vwi = 1:nviews,
  gtdata2d_size{vwi} = load(gtfile_trainsize{vwi});
  if ~isempty(gtfile_trainsize_cpr),
    gtdata2d_size{vwi} = AddCPRGTData(gtdata2d_size{vwi},gtfile_trainsize_cpr{vwi},lObj.labeledpos,vwi);
  end
end

gtdata2d_time = cell(1,nviews);
if ~isempty(gtfile_traintime),
  for vwi = 1:nviews,
    gtdata2d_time{vwi} = load(gtfile_traintime{vwi});
    if ~isempty(gtfile_traintime_cpr),
      gtdata2d_time{vwi} = AddCPRGTData(gtdata2d_time{vwi},gtfile_traintime_cpr{vwi},lObj.labeledpos,vwi);
    end
  end
end

annoterrdata2d = cell(1,nviews);
if ~isempty(annoterrfile),
  for vwi = 1:nviews,
    annoterrdata2d{vwi} = load(annoterrfile{vwi});
  end
end


% gtdata2d_cpr = cell(1,nviews);
% for vwi = 1:nviews,
%   fns = fieldnames(gtdata2d_size{vwi});
%   cprfns = fns(contains(fns,'cpr'));
%   for i = 1:numel(cprfns),
%     gtdata2d_cpr{vwi}.(cprfns{i}) = gtdata2d_size{vwi}.(cprfns{i});
%   end
% end
% gtdata_cpr = GT3DReconstruct(gtdata2d_cpr,incondinfo.gtinfo,'isshexp',true);
% for i = 1:numel(cprfns),
%   gtdata_size.(cprfns{i}) = gtdata_cpr.(cprfns{i});
% end

gtdata_size = GT3DReconstruct(gtdata2d_size,gtinfo,'isshexp',isshexp);
if ~isempty(gtfile_traintime),
  gtdata_time = GT3DReconstruct(gtdata2d_time,gtinfo,'isshexp',isshexp);
end
if ~isempty(annoterrfile),
  annoterrdata = GT3DReconstruct(annoterrdata2d,gtinfo,'isshexp',isshexp);
end

save(sprintf('%s3D_trainsize%s.mat',exptype,datestr(now,'yyyymmdd')),'-struct','gtdata_size');
if ~isempty(gtfile_traintime),
  save(sprintf('%s3D_traintime%s.mat',exptype,datestr(now,'yyyymmdd')),'-struct','gtdata_time');
end

if ~isempty(annoterrfile),
  save(sprintf('%s3D_AnnErrData%s.mat',exptype,datestr(now,'yyyymmdd')),'-struct','annoterrdata');
end

%%

FLYNUM2CALIB = '/groups/huston/hustonlab/flp-chrimson_experiments/fly2DLT_lookupTableStephen.csv';
fly2calib = readtable(FLYNUM2CALIB,...
  'Delimiter',',',...pwd
  'ReadVariableNames',false,...
  'HeaderLines',0);
fly2calib.Properties.VariableNames = {'fly' 'calibfile'};
fly2calibMap = containers.Map(fly2calib.fly,fly2calib.calibfile);
%%

PROJDIR = '/groups/branson/bransonlab/apt/experiments/res/sh_trn4523_gtcomplete_results';
RESFILE = 'cpr_gtres_v00_withanls_20180730.mat';
resfile = fullfile(PROJDIR,RESFILE);
res = load(resfile);
t = res.tGT;

%%
LBL = '/groups/branson/bransonlab/apt/experiments/data/sh_trn5017_20200121.lbl';
ld = loadLbl(LBL);
tLbl = Labeler.lblFileGetLabels(ld);
tLbl.vcd = ld.viewCalibrationData(tLbl.mov);
tLbl.movFile = ld.movieFilesAll(tLbl.mov,:);
flyid = cellfun(@parseSHfullmovie,tLbl.movFile);
assert(isequal(flyid(:,1),flyid(:,2)))
tLbl.flyID = flyid(:,1);

%%
[tfDLT,rpe] = compute_rperr_table(t,'pLbl',fly2calibMap);
%%
[tfDLT,rpe] = compute_rperr_table(tLbl,'p',fly2calibMap);
%%
rpemu = mean(rpe,3);

rpeflat = reshape(rpe,size(rpe,1),[]);
tfnan = isnan(rpeflat);
assert(isequal(tfnan,repmat(tfnan(:,1),1,size(tfnan,2))));
nOrtho = nnz(~tfnan(:,1) & ~tfDLT);
nDLT = nnz(~tfnan(:,1) & tfDLT);

ttlbig = sprintf('Mean reproj err (N=%d, %d/%d Ortho/DLT)',...
  nOrtho+nDLT,nOrtho,nDLT);
hfig = figure(12);
clf
pause(2);
hfig.Color = [1 1 1];
axs = createsubplots(1,5,[[.15 0];[.15 0]]);
%set(axs,'Color',[1 1 1]);
for ipt=1:5
  ax = axs(ipt);
  axes(ax);
  if ipt==1
    args = {'labels' {'Ortho' 'DLT'}...
      'labelorientation' 'horizontal'};
  else
    args = {'labels' {'' ''}};
  end
  boxplot(rpemu(:,ipt),tfDLT,args{:});
  grid on;
  if ipt==1
    
  else
    set(ax,'YTickLabel',[]);
  end
  set(ax,'FontSize',18);
  ttlstr = sprintf('pt%d',ipt);
  
  hTtl = title(ax,ttlstr,'interpreter','none','fontsize',18);
  hTtl.Units = 'pixels';
  hTtl.Position(2) = hTtl.Position(2)-1.6*hTtl.Extent(4);
end
linkaxes(axs);
ylim(axs(1),[-2 26]);
%ylabel(axs(1),'Reproj error (px)','interpreter','none','fontsize',24);
hfig.Name = ttlbig;
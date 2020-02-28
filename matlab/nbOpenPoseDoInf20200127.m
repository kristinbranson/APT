%%
mat = '/nrs/branson/al/cache/multitarget_bubble/openpose/view_0/apt_expt2/deepnet_resultsdo_inf_diags.mat';
lbls = '/groups/branson/bransonlab/apt/experiments/data/multitarget_bubble_expandedbehavior_20180425_FxdErrs_OptoParams20181126_dlstripped.lbl';
lbl = '/groups/branson/bransonlab/apt/experiments/data/multitarget_bubble_expandedbehavior_20180425_FxdErrs_OptoParams20181126.lbl';
%%
rd = load(mat);
ld = loadLbl(lbls);
rd.pks_with_score_cmpt = rd.pks_with_score_cmpt(:,:,:,[2 1 3]); % dumb
rd.info = squeeze(rd.info);
%% figure out the interesting examples
% initially did [rows,ipts] = find(rd.num_hm_peaks>1);
% but this includes a lot of extra exs where i think an upsampling artifact
% combined with our "max" method (template is nn but not diagonals) sees
% two kitty-corner neighboring pixels showing up twice.

[n,npts] = size(rd.conf);
tfinteresting = false(n,npts);
for i=1:n
for ipt=1:npts
  npk = rd.num_hm_peaks(i,ipt);
  npkcmp = nnz(rd.pks_with_score_cmpt(i,ipt,:,1)>=0);
  assert(npkcmp<=npk);
  
  if npk==2    
    pks = squeeze(rd.pks_with_score(i,ipt,1:npk,:));
    pks(:,1:2) = pks(:,1:2)+1;
    dpk = abs(pks(1,:)-pks(2,:));
    muscore = sum(pks(:,3))/2;
    if all(dpk(1:2)<=2.0) && dpk(3)/muscore<.05
      % no longer interesting, probable artifact peak
      assert(npkcmp==1);
    else
      tfinteresting(i,ipt) = true;
    end
  elseif npk>2
    tfinteresting(i,ipt) = true;
  else
    % npk<2, none
  end
end
end
%%
%[rows,ipts] = find(rd.num_hm_peaks>1);
[rows,ipts] = find(tfinteresting);
%tf = ipts>5;
%rows = rows(tf);
%ipts = ipts(tf);
nex = numel(rows);
fprintf(1,'%d exs\n',nex);
%%
FIGNUM = 11;
mdl = struct('nfids',1,'D',2); % for shape
hfig = figure(FIGNUM);
hfig.Color = [0 0 0];
clf;
axs = mycreatesubplots(1,3);
% iex/irow/ipt
% 2/47/11. pks had mildly better second option, looks like wide distro
% 3/184/12. smeared distro
% 7/1218/12. pk 2nd option good. smeared
% 10/1764/12. both pk and pkcmpt 2nd opt is right (way diff). significantly lower amp though. maybe context would help
% 15/689/13 2nd opt both pk and pkcmpt is right (way diff). very close amps.
% 18/1269/13. wgtcent did right thing, pk had number 1/2 opts flipped (both sig amp)
% 19/1283/13. pks had right 2nd opt with sig amp. pkcmpts only 1 
% 20/1448/13. both 2nd opts are right.
% 26/297/14. both pk and pkcmpts 1st opt is best! predloc must have failed b/c ncluster=1. XXXXXXXXXXX huh?
% 30/404/14. both 2nd opts were sig better. 2nd opt sig amp
% 42/148/15. pk opt2 better but maybe luck. pkcmpts only one opt
% 48/592/15. pk opt2 better (out of 4 opts) but very smeared not sure how that would be picked. all 4 pk opts comparable mag
% 49/616/15. both opt2s better but may be luck. close neighbor prob influencing.
% 58/1283/15. pk opt2 better almost same amp. pkcmps only one opt.
% 59/1296/15.  " 
% 62/1627/15. both opt2s are good. similar amps.
% 64/47/16. both opt2s slightly better (both orig and opt2s are bad tho). some luck maybe but opt2 sort of getting it more right
% 71/714/16. pk opt2 better almost same amp. pkcmps only one opt.
% 78/1136/16. pk opt2 better (out of 3). all 3 similar amps. pkcmps only one opt.
% 84/1369/16. locs was not opt1 of pkscmpt again! opt1 was better!! XXXXXXXXXXXXX
% 85/1474/16. smeary, pk opt1 and opt3 were better but pkcmpt opt1 chose closer to pk opt2
% 89/47/17. pk opt2 a hair better with nearly identical amp. opts are close tho 
% 92/244/17. opt2s better but sig lower amp.
% 93/337/16. pk opt1 best but pkcmpts opt1 spears it by ~3 px
% 100/950/17. both opt2s better with only slightly lower amps.
% 104/1342/17. smeary, pk opt4 better but no idea how to choose. 4 pkopts smilar amp kind of in medium vicinity. pkcmpts only 1 opt. wgtcent did ok here
% 106/1492/16. pk opt1 (out of 3) best here but gets smeared/pushed by ~3px. pkscmpt only one opt
%
% in about ~20/108 opt2 is better. 2 cases weirdness with opt1 being
% unexpected.
for iex=91:nex % xxxxxxxxxx 
  irow = rows(iex);
  ipt = ipts(iex);
  
  mft = rd.info(irow,:)+1;  
  Icell = {squeeze(rd.ims(irow,:,:))};
  
  npk = rd.num_hm_peaks(irow,ipt);
  locsGT = squeeze(rd.lbls(irow,ipt,:))+1;
  locsGT = locsGT'
  locs = squeeze(rd.locs(irow,ipt,:))+1;
  locs = locs'
  pks = squeeze(rd.pks_with_score(irow,ipt,1:npk,:));
  pks(:,1:2) = pks(:,1:2)+1;
  [~,idx] = sort(pks(:,3),1,'descend');
  pks = pks(idx,:)
  pkscmpt = squeeze(rd.pks_with_score_cmpt(irow,ipt,:,:));
  tfreal = pkscmpt(:,1)>=0;
  npkcmpt = nnz(tfreal);
  assert(npkcmpt<=npk);
  pkscmpt = pkscmpt(tfreal,:);
  pkscmpt(:,1:2) = pkscmpt(:,1:2)+1;
  [~,idx] = sort(pkscmpt(:,3),1,'descend');
  pkscmpt = pkscmpt(idx,:)
  
  Shape.vizSingle(Icell,locsGT(:)',1,mdl,'hax',axs(1,1),...
    'cmap',[1 0 0],...
    'p2',locs(:)','p2plotargs',...
     {'marker','+','markersize',10,'linewidth',1,'color',[1 0 0]});
   
  if npk>=2
    args = {'p2',pks(2,1:2),'p2plotargs',{'marker','+','markersize',10,'linewidth',2,'color',[1 0 0]}};
    if npk>=3
      warningNoTrace('Not including higher rows for pks');
    end
  else
    args = {};
  end
  Shape.vizSingle(Icell,pks(1,1:2),1,mdl,'hax',axs(1,2),...
    'cmap',[1 0 0],args{:});
  
  if npkcmpt>=2
    args = {'p2',pkscmpt(2,1:2),'p2plotargs',{'marker','+','markersize',10,'linewidth',2,'color',[1 0 0]}};
    if npkcmpt>=3
      warningNoTrace('Not including higher rows for pkcmpt');
    end
  else
    args = {};
  end
  Shape.vizSingle(Icell,pkscmpt(1,1:2),1,mdl,'hax',axs(1,3),...
    'cmap',[1 0 0],args{:});
  
  
  infostr = sprintf('iex/irow/pt=%d,%d,%d. %s: npk=%d, npkcmpt=%d',iex,irow,ipt,mat2str(mft),npk,npkcmpt);
  input(infostr);
end
classdef HPOptim
  
  % 1. Start with a split defn
  %    - outertrain vs outertest
  %    - outertrain has train/test xv splits
  % 2. Generate deltas/patches from current sPrm0.
  % 3. Fire xv jobs.
  % 4. When jobs all done. Compile results. Make plots. Select sPrmDelta.
  %   Update sPrm0. Goto 2. If sPrm0 is unchanged, stop. Take note of
  %   "downside" params that can be reduced without harm.
  % 5. Final train on full outertrain; apply to outertest.
  
  methods (Static)
    
    function aptClusterCmd
      cmd = sprintf('.../APTCluster.py -n 6 --outdir /groups/branson/bransonlab/apt/tmp/<bsubout> --bindate 20180710.feature.deeptrack --trackargs "tableFile /groups/branson/bransonlab/apt/tmp/<tableFile> tableSplitFile /groups/branson/bransonlab/apt/tmp/<tableSplitFile> paramFile /groups/branson/bransonlab/apt/tmp/<prmBase>" --prmpatchdir %s /groups/branson/bransonlab/apt/experiments/data/sh_trn4523_gt080618_made20180627_cacheddata.lbl xv',pchDir);
      fprintf(1,'Run:\n%s\n',cmd);
    end
    
    function genAndWritePchs(basePrmFile,pchDir,genPchsArgs)
      sPrm0 = loadSingleVariableMatfile(basePrmFile);
      s = HPOptim.genPchs(sPrm0,genPchsArgs{:});
      HPOptim.writePchDir(pchDir,s);
      fprintf(1,'Done creating pchdir: %s\n',pchDir);
    end
    
    function s = genPchs(sPrm0,varargin)
      [iterFac,radFac,midFac] = myparse(varargin,...
        'iterFac',1.4,...
        'radFac',1.6,...
        'midFac',1.5...
        );
      
      s = struct();
      
      T0 = sPrm0.ROOT.CPR.NumMajorIter;
      s.NumMajorIter_up = {sprintf('.ROOT.CPR.NumMajorIter=%d',round(iterFac*T0))};
      s.NumMajorIter_dn = {sprintf('.ROOT.CPR.NumMajorIter=%d',round(1/iterFac*T0))};
      
      K = sPrm0.ROOT.CPR.NumMinorIter;
      s.NumMinorIter_up = {sprintf('.ROOT.CPR.NumMinorIter=%d',round(iterFac*K))};
      s.NumMinorIter_dn = {sprintf('.ROOT.CPR.NumMinorIter=%d',round(1/iterFac*K))};
      
      FD = sPrm0.ROOT.CPR.Ferns.Depth;
      s.FernsDepth_up = {sprintf('.ROOT.CPR.Ferns.Depth=%d',FD+1)};
      s.FernsDepth_up2 = {sprintf('.ROOT.CPR.Ferns.Depth=%d',FD+2)};
      s.FernsDepth_dn = {sprintf('.ROOT.CPR.Ferns.Depth=%d',FD-1)};
      
      FTlo = sPrm0.ROOT.CPR.Ferns.Threshold.Lo;
      FThi = sPrm0.ROOT.CPR.Ferns.Threshold.Hi;
      FTmid = (FTlo+FThi)/2;
      FTrad = (FThi-FTlo)/2;
      s.FernThresholdMid_up = { ...
        sprintf('.ROOT.CPR.Ferns.Threshold.Lo=%.3f',FTmid); ...
        sprintf('.ROOT.CPR.Ferns.Threshold.Hi=%.3f',FTmid+2*FTrad); };
      s.FernThresholdMid_dn = { ...
        sprintf('.ROOT.CPR.Ferns.Threshold.Lo=%.3f',FTmid-2*FTrad); ...
        sprintf('.ROOT.CPR.Ferns.Threshold.Hi=%.3f',FTmid); };
      
      s.FernThresholdRad_up = { ...
        sprintf('.ROOT.CPR.Ferns.Threshold.Lo=%.3f',FTmid-2*FTrad); ...
        sprintf('.ROOT.CPR.Ferns.Threshold.Hi=%.3f',FTmid+2*FTrad); };
      s.FernThresholdRad_dn = { ...
        sprintf('.ROOT.CPR.Ferns.Threshold.Lo=%.3f',FTmid-0.5*FTrad); ...
        sprintf('.ROOT.CPR.Ferns.Threshold.Hi=%.3f',FTmid+0.5*FTrad); };
      
      RF0 = sPrm0.ROOT.CPR.Ferns.RegFactor;
      s.RegFactor_up = { sprintf('.ROOT.CPR.Ferns.RegFactor=%.3f',2*RF0) };
      s.RegFactor_dn = { sprintf('.ROOT.CPR.Ferns.RegFactor=%.3f',0.5*RF0) };
      
      TwoLMRad0 = sPrm0.ROOT.CPR.Feature.Radius;
      s.TwoLMRad_up = { sprintf('.ROOT.CPR.Feature.Radius=%.3f',radFac*TwoLMRad0) };
      s.TwoLMRad_dn = { sprintf('.ROOT.CPR.Feature.Radius=%.3f',1/radFac*TwoLMRad0) };
      
      TwoLMABRat0 = sPrm0.ROOT.CPR.Feature.ABRatio;
      s.TwoLMABRat_up = { sprintf('.ROOT.CPR.Feature.ABRatio=%.3f',radFac*TwoLMABRat0) };
      s.TwoLMABRat_dn = { sprintf('.ROOT.CPR.Feature.ABRatio=%.3f',1/radFac*TwoLMABRat0) };
      
      s.OneLM_lo = { '.ROOT.CPR.Feature.Type=''single landmark''' ; ...
        '.ROOT.CPR.Feature.Radius=30' };
      s.OneLM_md = { '.ROOT.CPR.Feature.Type=''single landmark''' ; ...
        '.ROOT.CPR.Feature.Radius=60' };
      s.OneLM_hi = { '.ROOT.CPR.Feature.Type=''single landmark''' ; ...
        '.ROOT.CPR.Feature.Radius=90' };
      
      FtrNGen0 = sPrm0.ROOT.CPR.Feature.NGenerate;
      s.FtrNGen_up = { sprintf('.ROOT.CPR.Feature.NGenerate=%d',round(radFac*FtrNGen0)) };
      s.FtrNGen_dn = { sprintf('.ROOT.CPR.Feature.NGenerate=%d',round(1/radFac*FtrNGen0)) };
      
      FtrNstd0 = sPrm0.ROOT.CPR.Feature.Nsample_std;
      s.FtrNstd_up = { sprintf('.ROOT.CPR.Feature.Nsample_std=%d',round(radFac*FtrNstd0)) };
      s.FtrNstd_dn = { sprintf('.ROOT.CPR.Feature.Nsample_std=%d',round(1/radFac*FtrNstd0)) };
      
      FtrNcor0 = sPrm0.ROOT.CPR.Feature.Nsample_cor;
      s.FtrNcor0_up = { sprintf('.ROOT.CPR.Feature.Nsample_cor=%d',round(midFac*FtrNcor0)) };
      s.FtrNcor0_dn = { sprintf('.ROOT.CPR.Feature.Nsample_cor=%d',round(1/midFac*FtrNcor0)) };
      
      NrepTrn0 = sPrm0.ROOT.CPR.Replicates.NrepTrain;
      s.NrepTrn_up = { sprintf('.ROOT.CPR.Replicates.NrepTrain=%d',round(iterFac*NrepTrn0)) };
      s.NrepTrn_dn = { sprintf('.ROOT.CPR.Replicates.NrepTrain=%d',round(1/iterFac*NrepTrn0)) };
      
      NrepTrk0 = sPrm0.ROOT.CPR.Replicates.NrepTrack;
      s.NrepTrk_up = { sprintf('.ROOT.CPR.Replicates.NrepTrack=%d',round(iterFac*NrepTrk0)) };
      s.NrepTrk_dn = { sprintf('.ROOT.CPR.Replicates.NrepTrack=%d',round(1/iterFac*NrepTrk0)) };
      
      PtJitFac0 = sPrm0.ROOT.CPR.Replicates.PtJitterFac;
      s.PtJitFac_up = { sprintf('.ROOT.CPR.Replicates.PtJitterFac=%d',2*PtJitFac0) };
      s.PtJitFac_dn = { sprintf('.ROOT.CPR.Replicates.PtJitterFac=%d',round(PtJitFac0/1.2)) };
      
      s.BoxJit_lo = {...
        '.ROOT.CPR.Replicates.DoBBoxJitter=1'; ...
        '.ROOT.CPR.Replicates.AugJitterFac=12' };
      s.BoxJit_md = {...
        '.ROOT.CPR.Replicates.DoBBoxJitter=1'; ...
        '.ROOT.CPR.Replicates.AugJitterFac=16' };
      s.BoxJit_hi = {...
        '.ROOT.CPR.Replicates.DoBBoxJitter=1'; ...
        '.ROOT.CPR.Replicates.AugJitterFac=24' };
      
      AugUseFF0 = sPrm0.ROOT.CPR.Replicates.AugUseFF;
      s.AugUseFF_flp = { sprintf('.ROOT.CPR.Replicates.AugUseFF=%d',~AugUseFF0) };
      
      PruneSig0 = sPrm0.ROOT.CPR.Prune.DensitySigma;
      s.PruneSig_up = { sprintf('.ROOT.CPR.Prune.DensitySigma=%.3f',2*PruneSig0) };
      s.PruneSig_dn = { sprintf('.ROOT.CPR.Prune.DensitySigma=%.3f',1/2*PruneSig0) };
    end
    
    function writePchDir(pchdir,s)
      if exist(pchdir,'dir')==0
        fprintf(1,'Making pch dir: %s\n',pchdir);
        [succ,msg] = mkdir(pchdir);
        if ~succ
          error('Failed to create dir: %s\n',msg);
        end
      end
      
      fns = fieldnames(s);
      for f=fns(:)',f=f{1}; %#ok<FXSET>
        fname = fullfile(pchdir,[f '.m']);
        cellstrexport(s.(f),fname);
        fprintf(1,'Wrote %s\n',fname);
      end
    end
    
    function printPrmPchDir(pchdir)
      dd = dir(fullfile(pchdir,'*.m'));
      nn = {dd.name}';
      for n=nn(:)',n=n{1}; %#ok<FXSET>
        fname = fullfile(pchdir,n);
        [~,fnameS] = fileparts(fname);
        fprintf(1,'%s\n',fnameS);
        l = readtxtfile(fname);
        fprintf(1,'%s\n',l{:});
      end
    end
    
    function printPrmPchs(s)
      fns = fieldnames(s);
      for f=fns(:)',f=f{1}; %#ok<FXSET>
        fprintf(1,'%s\n',f);
        disp(s.(f));
        fprintf(1,'\n');
      end
    end
    
    function [xverr,pchNames,xvres] = loadXVres(pchDir,xvDir,xvPat,varargin)
      % xvPat: sprintf-pat given a pchname to form xv-results-filename
      % 
      % xverr: [nXVxnptsxnpch] (npts==nPhysPtsxnView)
      % pchNames: [npch] cellstr
      
      xvbase = myparse(varargin,...
        'xvbase',''); % if supplied, load 'base' xv results with this name
      tfBase = ~isempty(xvbase);
      
      % Get patches
      dd = dir(fullfile(pchDir,'*.m'));
      pchs = {dd.name}';
      if tfBase
        pchs = [xvbase; pchs];
      end
      npch = numel(pchs);      

      %% Load res
      xvres = cell(npch,1);
      pchNames = cell(npch,1);
      for i=1:npch
        if tfBase && i==1
          xvresfnameS = xvbase;
          pchS = 'NOPATCH';
        else
          [~,pchS,~] = fileparts(pchs{i});
          pat = sprintf(xvPat,pchS);
          dd = dir(fullfile(xvDir,pat));
          assert(isscalar(dd));
          xvresfnameS = dd.name;
        end
        xvresfname = fullfile(xvDir,xvresfnameS);
        assert(exist(xvresfname,'file')>0);        
        xvres{i,1} = load(xvresfname);
        pchNames{i} = pchS;
        fprintf(1,'%d\n',i);
      end
%       xvres0 = {...
%         load(fullfile(XVRESDIR,'xv_tMain4523_tMain4523_split3_easy_prmMain0_20180710T103811.mat')) ...
%         load(fullfile(XVRESDIR,'xv_tMain4523_tMain4523_split3_hard_prmMain0_20180710T104734.mat'))};
%       xvres = [xvres0; xvres];
%       pchNames = [{'base'}; pchNames];
      
      %%
%       assert(npch+1==size(xvres,1));
      nXV = height(xvres{1}.xvRes);
      npts = size(xvres{1}.xvRes.dGTTrk,2);
      fprintf(1,'nXV=%d, npts=%d, %d patches.\n',nXV,npts,npch);
      xverr = nan(nXV,npts,npch);
      for i=1:npch
        xverr(:,:,i) = xvres{i}.xvRes.dGTTrk;
      end
    end
    
    function tblres = pchScores(xverr,pchNames,varargin)
      [ipchBase,ptilesImprove,nptsImproveThresh] = myparse(varargin,...
        'ipchBase',1, ... % reference/base patch
        'ptilesImprove',[60 90], ...
        'nptsImproveThresh',8 ...
        );
      
      [nXV,npts,npch] = size(xverr);
      
      %% normalized err; for each pt, scale by the median 'base' err
      xverrbasemedn = median(xverr(:,:,ipchBase),1);
      xverrnorm = xverr./xverrbasemedn;
      
      assert(all(xverrnorm(:)>0));
      nptiles = numel(ptilesImprove);
      errptl = prctile(xverrnorm,ptilesImprove,1); % [nptiles x npts x npch]
      
      % compute fractional change in ptile values from base, per pt
      % 0=>no change in ptile
      % pos=>reduced err
      % neg=>increased err
      scorepts = (errptl(:,:,ipchBase)-errptl)./errptl(:,:,ipchBase);
      szassert(scorepts,[nptiles npts npch]);      
      
      % pt is improved if it is improved for all ptiles
%      ptimproved = scorepts>0;
%      nptimprove = squeeze(sum(sum(ptimproved,1),2)); %[npch]
      ptimproved = all(scorepts>0,1); % every ptile
      ptimproved = squeeze(ptimproved)'; 
      szassert(ptimproved,[npch npts]);
      ptimprovedfull = scorepts>0;
      
      % net score: score averaged over all pts, ptiles, per patch
      % ptiles that have larger relative fluctuations get more weight
      score = sum(sum(scorepts,1),2)/npts/nptiles*100;
      score = squeeze(score); % [npch x 1]. %age relative change in ptile for each pt
            
      nptimprove = sum(ptimproved,2);
      nptimprovedfull = squeeze(sum(sum(ptimprovedfull,1),2));
      pch = pchNames;
      tblres = table(score,nptimprove,nptimprovedfull,pch);
      [~,idx] = sort(score,'descend');
      tblres = tblres(idx,:);
    end
    
    function sPrm = readApplyPatch(sPrm,pchfile,varargin)
      verbose = myparse(varargin,...
        'verbose',true);
      
      pchs = readtxtfile(pchfile,'discardemptylines',true);
      npch = numel(pchs);
%       fprintf(1,'Read parameter patch file %s. %d patches.\n',paramPatchFile,...
%         npatch);  
      for ipch=1:npch
        pch = pchs{ipch};
        pch = ['sPrm' pch ';']; %#ok<AGROW>
        tmp = strsplit(pch,'=');
        pchlhs = strtrim(tmp{1});
        if verbose
          fprintf(1,'  patch %d: %s\n',ipch,pch);
          fprintf(1,'  orig (%s): %s\n',pchlhs,evalc(pchlhs));
        end
        eval(pch);
        if verbose
          fprintf(1,'  new (%s): %s\n',pchlhs,evalc(pchlhs));
        end
      end
    end
    
    function genNewPrmFile(basePrmFile,newPrmFile,pchDir,pchSel)
      % Create/save a new parameter file
      %
      % basePrmFile: char, base/starting parameter struct
      % newPrmFile: char, save new params to this file
      % pchDir: patch dir
      % pchSel: cellstr, patches in pchDir to apply
      
      sPrm = loadSingleVariableMatfile(basePrmFile);
      fprintf(1,'Loaded base parameters from %s.\n',basePrmFile);
      sPrm0 = sPrm;
      
      npchsel = numel(pchSel);
      for ipchsel=1:npchsel
        pchfile = fullfile(pchDir,pchSel{ipchsel});
        if ~strcmp(pchfile(end-1:end),'.m')
          pchfile = [pchfile '.m']; %#ok<AGROW>
        end
        sPrm = HPOptim.readApplyPatch(sPrm,pchfile);        
      end
      
      leap.structdiff(sPrm0,sPrm);
      
      if exist(newPrmFile,'file')>0
        error('File ''%s'' exists.',newPrmFile);
      end
      save(newPrmFile,'-mat','sPrm');
      fprintf(1,'Wrote new parameter file to %s.\n',newPrmFile);
    end
  end
    
    %     function hpoptimxv(lObj,xvTbl,xvSplt)
    %       assert(islogical(xvSplt) && ismatrix(xvSplt) && size(xvSplt,1)==height(xvTbl));
    %       error('Expected split definition to be a logical matrix with %d rows.\n',...
    %         height(xvTbl));
    %
    %       kfold = size(xvSplt,2);
    %       wbObj = WaitBarWithCancelCmdline;
    %       xvArgs = {'kfold' kfold 'wbObj' wbObj 'tblMFgt' xvTbl 'partTst' xvSplt};
    %
    %       % base case
    %       sPrm0 = lObj.trackGetParams();
    %       lObj.trackCrossValidate(xvArgs{:});
    %       xv0 = lObj.xvResults;
    %
    %       while 1
    %         % generate patches
    %         paramDels = HPOptim.genPrmDeltas(sPrm0);
    %         xvDels = structfun(@(x)lclRunWithPrm(lObj,x),paramDels,'uni',0);
    %         % xvDels is [nDel x 1] cell of xv results
    %
    %         % compute scores
    %
    %         % look where going down on some params wouldn't have hurt
    %
    %         % select those deltas where
    %         % - >= nptsThresh (8) pts were improved
    %         sPrm0 = lclGenNewParams();
    %         lObj.trackSetParams(sPrm0);
    %
    %         % save state
    %
    %         % stopping criterion
    %         if stop
    %           break;
    %         end
    %       end
    %
    %       % lObj has latest params
    %     end
    
    %
    %
    %
    % if tfPPatch
    %       [~,paramPatchFileS,~] = fileparts(paramPatchFile);
    %       patches = readtxtfile(paramPatchFile);
    %       npatch = numel(patches);
    %       fprintf(1,'Read parameter patch file %s. %d patches.\n',paramFile,...
    %         npatch);
    %       sPrm = lObj.trackGetParams();
    %       for ipch=1:npatch
    %         pch = patches{ipch};
    %         pch = ['sPrm' pch ';']; %#ok<AGROW>
    %         tmp = strsplit(pch,'=');
    %         pchlhs = strtrim(tmp{1});
    %         fprintf(1,'  patch %d: %s\n',ipch,pch);
    %         fprintf(1,'  orig (%s): %s\n',pchlhs,evalc(pchlhs));
    %         eval(pch);
    %         fprintf(1,'  new (%s): %s\n',pchlhs,evalc(pchlhs));
    %       end
    %       lObj.trackSetParams(sPrm);
    %       outfileBase = [outfileBase '_' paramPatchFileS];
    %     end
    %     outfileBase = [outfileBase '_' datestr(now,'yyyymmddTHHMMSS')];
    %
    %
    %     savestuff = struct();
    %     savestuff.sPrm = lObj.trackGetParams();
    %     savestuff.xvArgs = xvArgs;
    %     savestuff.xvRes = lObj.xvResults;
    %     savestuff.xvResTS = lObj.xvResultsTS; %#ok<STRNU>
    %     outfile = fullfile(lblP,[outfileBase '.mat']);
    %     fprintf('APTCluster: saving xv results: %s\n',outfile);
    %     save(outfile,'-mat','-struct','savestuff');
    % end
  
  
end
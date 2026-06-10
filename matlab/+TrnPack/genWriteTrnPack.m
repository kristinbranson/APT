function [slbl,tp,locg,ntgtstot] = genWriteTrnPack(lObj,dmc,varargin)
  % Generate training package. Write contents (raw images and keypt
  % jsons) to packdir.

  [writeims,writeimsidx,~,~,verbosejson,tblsplit,~,...
    cocoformat,jsonfilename] = myparse(varargin,...
    'writeims',true, ...
    'writeimsidx',[], ... % (opt) DEBUG ONLY
    'trainConfigName','',...
    'strippedlblname',[], ... % (reqd) short filename for stripped lbl
    'verbosejson',false, ...
    'tblsplit', [], ...  % tbl with fields .mov, .frm, .split
                   ...  % all double/numeric and 1-based
    'view',nan, ... % currently unused
    'cocoformat',false, ... % added by KB 20250522, export coco json file format
    'jsonfilename',''... % added by KB 20250522, name (not path) of file to export to
    );

  tfsplitsprovided = ~isempty(tblsplit);

  packdir = dmc.dirProjLnx;
  if isempty(jsonfilename),
    jsonoutf = dmc.trainLocLnx;
  else
    jsonoutf = fullfile(packdir,jsonfilename);
  end

  if exist(packdir,'dir')==0
    mkdir(packdir);
  end

  tObj = lObj.tracker;
  tObj.setAllParams(lObj.trackGetTrainingParams()); % does not set skel, flipLMEdges

  slbl_orig = tObj.trnCreateStrippedLbl();
  slbl = compressStrippedLbl(slbl_orig,'ma',true);
  [~,jslbl] = jsonifyStrippedLbl(slbl);

  % Commenting this out b/c the the predicate is always false.
  % -- ALT, 2025-10-08
  % if (~cocoformat) && (strcmp(DeepModelChainOnDisk.configFileExt,'.lbl') || DeepModelChainOnDisk.gen_strippedlblfile),
  %   sfname = dmc.lblStrippedLnx;
  %   save(sfname,'-mat','-v7.3','-struct','slbl');
  %   fprintf(1,'Saved %s\n',sfname);
  % end

  if (~cocoformat) && (strcmp(DeepModelChainOnDisk.configFileExt,'.json')),
    % KB 20250527 -- does the json file get saved twice?
    TrnPack.hlpSaveJson(jslbl,DeepModelChainOnDisk.getCheckSingle(dmc.trainConfigLnx));
  end

  % use stripped lbl trackerData instead of tObj, as we have called
  % addExtraParams etc.

  tdata2 = slbl.trackerData{2};
  netmode2 = tdata2.trnNetMode;
  % For top-down nets, whether trnNetMode from .trackerData{2} or
  % .trackerData{1} is used should be irrelevant to isObjDet.
  isObjDet = netmode2.isObjDet;
  % Again, getting sPrmBBox and sPrmLossMask from trackerData{2} should
  % be fine for top-down nets. trackerData{1} should match for these
  % params.
  sPrmMA = tdata2.sPrmAll.ROOT.MultiAnimal;
  sPrmBBox = sPrmMA.Detect.BBox;
  sPrmLossMask = sPrmMA.LossMask;
  tp = TrnPack.aggregateLabelsAddRoi(lObj,isObjDet,sPrmBBox,sPrmLossMask);

  % add splits to tp
  nmov = numel(tp);
  if tfsplitsprovided

    % split verification
    tblsplitMF = tblsplit(:,{'mov' 'frm'});
    assert(height(tblsplitMF)==height(unique(tblsplitMF)),...
      'Split table contains one or more duplicated (movie,frame) pairs.');
    allsplits = tblsplit.split;
    assert(isequal(round(allsplits),allsplits));
    assert(all(allsplits>0));
    nsplits = max(allsplits);
    assert(isequal(unique(allsplits),(1:nsplits)'));

    tfsplitrowused = false(height(tblsplit),1);
    for imov=1:nmov
      tfmov = tblsplit.mov==imov;
      idxmov = find(tfmov);
      fsplit = tblsplit.frm(tfmov);
      jsplit = tblsplit.split(tfmov);

      [tf,loc] = ismember(tp(imov).frm,fsplit);
      split = zeros(size(tp(imov).frm),Labels.CLS_SPLIT());
      nMissing = nnz(~tf);
      if nMissing>0
        warningNoTrace('Movie %d: %d labeled rows not found in split table. These will be added to split 1.',...
          imov,nMissing);
        split(~tf) = 1;
      end

      split(tf) = jsplit(loc);
      tfsplitrowused(idxmov(loc)) = true;
      tp(imov).split = split;
    end

    nExtra = nnz(~tfsplitrowused);
    if nExtra>0
      warningNoTrace('Ignoring %d unused/extraneous rows in split table.',nExtra);
    end

    fprintf(1,'Split summary:\n');
    summary(categorical(allsplits(tfsplitrowused)));
  else
    for imov=1:nmov
      tp(imov).split = ones(size(tp(imov).frm),Labels.CLS_SPLIT());
    end
  end

  if lObj.gtIsGTMode
    movinfo = lObj.movieInfoAllGT;
  else
    movinfo = lObj.movieInfoAll;
  end
  isma = lObj.maIsMA;
  if isma
    locg = TrnPack.genLocs(tp,movinfo);
    if writeims
      if isempty(writeimsidx)
        writeimsidx = 1:numel(locg);
      end

      TrnPack.writeims(locg(writeimsidx),packdir);
    end
  else
    locg = TrnPack.genLocsSA(slbl_orig,tblsplit);
    if writeims
      if isempty(writeimsidx)
        writeimsidx = 1:numel(locg);
      end

      TrnPack.writeimsSA(locg(writeimsidx),packdir,slbl_orig.preProcData_I);
    end
  end

  if verbosejson
    % trnpack: one row per mov
    %jsonoutf = 'trnpack.json';
    TrnPack.hlpSaveJson(tp,jsonoutf);
  end

%       % loc: one row per labeled tgt
%       jsonoutf = 'loc0.json';
%       TrnPack.hlpSaveJson(loc,packdir,jsonoutf);

  % loc: one row per frm
  %jsonoutf = 'loc.json';
  s = struct();
  s.movies = lObj.movieFilesAllGTaware;
  if tfsplitsprovided
    s.splitnames = arrayfun(@(x)sprintf('split%02d',x),1:nsplits,'uni',0);
  else
    s.splitnames = {'trn'};
  end
  s.locdata = locg;

  if cocoformat,
    % convert to COCO format
    s = TrnPack.ConvertTrnPackLocToCOCO(s,packdir,'skeleton',lObj.skeletonEdges,'keypoint_names',lObj.skelNames,'isma',isma);
  else
    s.kpt_info = struct();
    s.kpt_info.skeleton=lObj.skeletonEdges;
    s.kpt_info.keypoint_names = lObj.skelNames;
  end

  TrnPack.hlpSaveJson(s,jsonoutf);

  ntgtstot = sum([locg.ntgt]);

%       % loccc: one row per cluster
%       jsonoutf = 'locclus.json';
%       TrnPack.hlpSaveJson(loccc,packdir,jsonoutf);
end % function

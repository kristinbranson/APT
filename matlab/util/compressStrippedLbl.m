function s = compressStrippedLbl(s,varargin)
  iscfg = isfield(s,'cfg');
  if iscfg,
    isMA = s.cfg.MultiAnimal;
  end

  FLDS = {'cfg' 'projname' 'projectFile' 'projMacros' 'cropProjHasCrops' ...
    'trackerClass' 'trackerData'};
  if iscfg,
    CFG_GLOBS = {'Num' 'MultiAnimal' 'HasTrx'};
  else
    FLDS = setdiff(FLDS,{'cfg'});
  end
  TRACKERDATA_FLDS = {'sPrmAll' 'trnNetMode' 'trnNetTypeString'};
  if iscfg,
    if isMA
      GLOBS = {'movieFilesAll' 'movieInfoAll' 'trxFilesAll'};
      FLDSRM = {'projMacros'};
    else
      GLOBS = {'labeledpos' 'movieFilesAll' 'movieInfoAll' 'trxFilesAll'};
      FLDSRM = { ... % 'movieFilesAllCropInfo' 'movieFilesAllGTCropInfo' ...
        'movieFilesAllHistEqLUT' 'movieFilesAllGTHistEqLUT'};
    end
    fldscfg = fieldnames(s.cfg);
    fldscfgkeep = fldscfg(startsWith(fldscfg,CFG_GLOBS));
    s.cfg = structrestrictflds(s.cfg,fldscfgkeep);
  else
    GLOBS = {};
    FLDSRM = {};
  end

  for i=1:numel(s.trackerData)
    if ~isempty(s.trackerData{i})
      s.trackerData{i} = structrestrictflds(s.trackerData{i},...
                                            TRACKERDATA_FLDS);
    end
  end

  flds = fieldnames(s);
  fldskeep = flds(startsWith(flds,GLOBS));
  fldskeep = [fldskeep(:); FLDS(:)];
  fldskeep = setdiff(fldskeep, FLDSRM);
  s = structrestrictflds(s,fldskeep);
end % function

classdef Lbl
  methods (Static)
    function s = compressStrippedLbl(s)
      CFG_GLOBS = {'Num'};
      FLDS = {'cfg' 'projname' 'projMacros' 'movieInfoAll' 'cropProjHasCrops' ...
        'trackerClass' 'trackerData'};
      GLOBS = {'labeledpos' 'movieFilesAll' 'trxFilesAll' 'preProcData'};
      
      fldscfg = fieldnames(s.cfg);      
      fldscfgkeep = fldscfg(startsWith(fldscfg,CFG_GLOBS));
      s.cfg = structrestrictflds(s.cfg,fldscfgkeep);
      
      flds = fieldnames(s);
      fldskeep = flds(startsWith(flds,GLOBS));
      fldskeep = [fldskeep(:); FLDS(:)];
      s = structrestrictflds(s,fldskeep);
    end
  end
end
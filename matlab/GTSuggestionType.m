classdef GTSuggestionType
  % Currently unused, see GTSetNumFramesType.m for current (very similar)
  % treatment
 
  enumeration
    RANDOM % each MFT row drawn equally likely
    BALANCEDMOVIE % number of rows per movie approx equal
    BALANCEDTARGET % number of rows per target/movie approx equal
  end
      
  methods
    function tblMFT = sampleMFTTable(obj,tblMFT,nsamp)
      % Draw/sample rows of a tblMFT
      % 
      % nsamp: total number of rows desired in output
      
      nrow = height(tblMFT);
      if nsamp>nrow
        warningNoTrace('Table has too few rows for desired sampling.');
        nsamp = nrow;
      end
      
      switch obj
        case GTSuggestionType.RANDOM
          g = repmat(ones,nrow,1);
        case GTSuggestionType.BALANCEDMOVIE
          assert(isa(tblMFT.mov,'MovieIndex'));
          g = tblMFT.mov.get();
        case GTSuggestionType.BALANCEDTARGET
          assert(isa(tblMFT.mov,'MovieIndex'));
          iMovAbs = tblMFT.mov.get();
          assert(all(iMovAbs>0));
          iMovAbs = uint64(iMovAbs);
          assert(all(tblMFT.iTgt>0));
          iTgt = uint64(tblMFT.iTgt);
          maxMov = max(iMovAbs);
          maxTgt = max(iTgt);
          MAXKEY = intmax('uint64');
          assert(maxMov*maxTgt<MAXKEY);
          g = (iMovAbs-1)*maxTgt + iTgt;
        otherwise
          assert(false);
      end
      
      idx = GTSuggestionType.balancedSample(nrow,nsamp,g);
      tblMFT = tblMFT(idx,:);
      tblMFT = MFTable.sortCanonical(tblMFT);
    end
  end
  
  methods (Static)    
    function idx = balancedSample(nrow,nsamp,g)
      assert(iscolumn(g) && numel(g)==nrow);
      assert(nsamp<=nrow);
      
      grpUn = unique(g);
      ngrp = numel(grpUn);
      nsampPerGrp = ceil(nsamp/ngrp);
      idx = zeros(0,1);
      for igrp=1:ngrp
        idxGrp = find(g==grpUn(igrp)); % row indices
        ntmp = numel(idxGrp);
        idxGrpSamp = idxGrp(randsample(ntmp,min(ntmp,nsampPerGrp)));
        idx = [idx; idxGrpSamp(:)]; %#ok<AGROW>
      end
    end    
  end
  
end
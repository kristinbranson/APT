classdef GTSuggestionType
  
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
          g = tblMFT.mov;
        case GTSuggestionType.BALANCEDTARGET
          assert(all(tblMFT.mov>0));
          assert(all(tblMFT.iTgt>0));
          maxMov = max(tblMFT.mov);
          maxTgt = max(tblMFT.iTgt);
          MAXKEY = intmax('uint64');
          assert(maxMov*maxTgt<MAXKEY);
          g = (tblMFT.mov-1)*maxTgt + tblMFT.iTgt;
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
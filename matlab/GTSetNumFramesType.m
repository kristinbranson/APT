classdef GTSetNumFramesType
  enumeration
    Total
    PerMovie
    PerTarget
  end
  
  methods 
    function tblsamp = sample(obj,tbl,n)
      % Sample n frames from an MFTtable
      %
      % tbl: input MFTTable
      %
      % tblsamp: output MFTTable, sampled rows from tbl
      
      nBase = height(tbl);
      switch obj
        case GTSetNumFramesType.Total
          if n>nBase
            warningNoTrace('Only %d GT rows are available.',nBase);
            tblsamp = tbl;
          else
            iSamp = randsample(nBase,n);
            tblsamp = tbl(iSamp,:);
          end
        case GTSetNumFramesType.PerMovie
          gC = categorical(tbl.mov);
          iSamp = GTSetNumFramesType.balancedsamp(gC,n,'movie');
          tblsamp = tbl(iSamp,:);
        case GTSetNumFramesType.PerTarget
          gC = categorical(tbl.mov).*categorical(tbl.iTgt);
          iSamp = GTSetNumFramesType.balancedsamp(gC,n,'target');
          tblsamp = tbl(iSamp,:);
        otherwise
      end
    end
  end
  methods (Static)
    function isamp = balancedsamp(gC,nsamp,catname)
      % gC: categorical grouping vector
      % nsamp: number of rows to sample for each category
      % catname: string describing category, for warning msgs
      %
      % isamp: selected indices into gC
      
      gC = removecats(gC);
      isamp = zeros(0,1);
      cats = categories(gC);
      for iCat=1:numel(cats)
        tfCat = gC==cats(iCat);
        idxThisCat = find(tfCat);
        nThisCat = numel(idxThisCat);
        if nsamp>nThisCat
          warningNoTrace('Only %d GT rows are available in %s.',nThisCat,catname);
          isamp = [isamp; idxThisCat]; %#ok<AGROW>
        else
          isamp = [isamp; idxThisCat(randsample(nThisCat,nsamp))]; %#ok<AGROW>
        end
      end
    end
  end
end
classdef MovieIndex < int32
  % A MovieIndex is a unique identifier for a movie(set) in an APT project.
  %
  % For a long time, this purpose was served by a single positive (row) 
  % index into labeler.movieFilesAll ('iMov'). With the addition GT mode 
  % and the GT movielist, a single positive index is no longer sufficient.
  %
  % Indices to non-GT movies seem to be encoded as positive int32's, indices to
  % GT movies are encoded as negative int32's.  --ALT, 2024-05-01
  methods 
    
    function obj = MovieIndex(varargin)
      % MovieIndex(iMovSgnedArr)
      % MovieIndex(iMovArr,gt) % gt is logical scalar
      if nargin==1
        iMovSgned = varargin{1};
      else
        iMov = varargin{1};
        assert(all(iMov>=0));
        gt = varargin{2};
        assert(isscalar(gt) && islogical(gt));
        if gt
          iMovSgned = -iMov;
        else
          iMovSgned = iMov;
        end
      end
      %assert(~any(iMovSgned==0));
      obj@int32(iMovSgned);
    end
    
    function [iMov,gt] = get(objArr)
      iMov = abs(objArr);
      gt = objArr<0;
    end
        
    function [tf,tfGT] = isConsistentSet(objArr)
      % tf: true if elements of objArr are either all GT or all notGT
      assert(~isempty(objArr));
      tfPos = objArr>0;
      tfNeg = objArr<0;
      tf = all(tfPos) || all(tfNeg);
      tfGT = objArr(1)<0;
    end
    
    function i = id32(objArr)
      i = int32(objArr);
    end
    
  end
  
end
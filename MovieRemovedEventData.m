classdef MovieRemovedEventData < event.EventData
  properties
    % containers.Map. iMovOrig2New(iMovOrig) gives iMovNew, the new/updated 
    % (after movie removal) movie index for iMovOrig. If movie iMovOrig is 
    % no longer present, iMovNew will equal 0.
    iMovOrig2New 
  end
  methods
    function obj = MovieRemovedEventData(iMovRmSigned,nMovOrigReg,...
                                         nMovOrigGT)
      % iMovRmSigned: positive/negative for reg/gt movies
      % nMovOrigReg: original number of regular movies
      % nMovOrigGT: " GT movies
      
      assert(isscalar(iMovRmSigned) && iMovRmSigned~=0);
      assert(nMovOrigReg>0);
      assert(nMovOrigGT>0);
      
      origIdxs = [1:nMovOrigReg -1:-1:-nMovOrigGT];
      if iMovRmSigned>0
        newIdxs = [1:iMovRmSigned-1 0 iMovRmSigned:nMovOrigReg-1 ...
                  -1:-1:-nMovOrigGT];
      else
        newIdxs = [1:nMovOrigReg ...
                   -1:-1:iMovRmSigned+1 0 iMovRmSigned:-1:-nMovOrigGT+1];
      end
      obj.iMovOrig2New = containers.Map(origIdxs,newIdxs);
    end
  end
end
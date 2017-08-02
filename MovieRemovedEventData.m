classdef MovieRemovedEventData < event.EventData
  properties
    % [nOrigMovx1] vector. iMovOrig2New(iMovOrig) gives iMovNew, the 
    % new/updated (after movie removal) movie index for iMovOrig. If movie 
    % iMovOrig is no longer present, iMovNew will equal 0.
    iMovOrig2New 
  end
  methods
    function obj = MovieRemovedEventData(iMovRemoved,nMovOrig)
      m = zeros(nMovOrig,1);
      m([1:iMovRemoved-1 iMovRemoved+1:nMovOrig]) = 1:nMovOrig-1;
      obj.iMovOrig2New = m;
    end
  end
end
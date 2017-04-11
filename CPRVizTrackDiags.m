classdef CPRVizTrackDiags < handle
  properties
    hFig % CPRVizTrackDiagsGUI
    gdata % handles
    
    lObj % Labeler obj
    tObj % Tracker obj
    rcObj % RegressorCascade obj
    
    nRep % number of replicates
    tMax % maximum major iter
    uMax % max minor iter
  end
  properties (SetObservable)    
    iRep % replicate index
    t % major iter
    u % minor iter
  end
  methods
    function set.iRep(obj,v)
      if v>=1 && v<=obj.nRep %#ok<MCSUP>
        obj.iRep = v;
      end
    end
    function set.t(obj,v)
      if v>=1 && v<=obj.tMax %#ok<MCSUP>
        obj.t = v;
      end
    end
    function set.u(obj,v)
      if v>=1 && v<=obj.uMax %#ok<MCSUP>
        obj.u = v;
      end
    end
  end
  
  methods
    function obj = CPRVizTrackDiags(lObj,hFig)
      obj.hFig = hFig;
      
      assert(isa(lObj,'Labeler'));      
      obj.lObj = lObj;
      obj.tObj = lObj.tracker;
      obj.rcObj = lObj.tracker.trnResRC;
      
      obj.nRep = lObj.tracker.sPrm.TestInit.Nrep;
      obj.tMax = lObj.tracker.trnResRC.nMajor;
      obj.uMax = lObj.tracker.trnResRC.nMinor;
    end
    function init(obj)
      obj.gdata = guidata(obj.hFig);

      obj.iRep = 1;
      obj.t = 1;
      obj.u = 1;      
    end
  end
  methods    
    function [ipts,ftrtype] = getLandmarksUsed(obj)
      % f: [nMinor x M x nUse]
      rc = obj.rcObj;
      [ipts,ftrtype] = rc.getLandmarksUsed(obj.t);
    end
  end  
end
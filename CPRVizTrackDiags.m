classdef CPRVizTrackDiags < handle
  properties
    isinit % true during initialization
    
    hFig % CPRVizTrackDiagsGUI
    gdata % handles
    
    lObj % Labeler obj
    tObj % Tracker obj
    rcObj % RegressorCascade obj
    
    hViz % [MxnUse] cell array of handles for visualization 
  end
  properties (SetObservable)    
    iRep % replicate index
    t % major iter
    u % minor iter
  end
  properties (Dependent,SetAccess=private)
    nRep % number of replicates
    tMax % maximum major iter
    uMax % max minor iter    
    M % number of ferns
    metaNUse % either 1 or 2 depending on feature.metatype
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
    function v = get.nRep(obj)
      v = obj.tObj.sPrm.TestInit.Nrep;
    end
    function v = get.tMax(obj)
      v = obj.rcObj.nMajor;
    end    
    function v = get.uMax(obj)
      v = obj.rcObj.nMinor;
    end
    function v = get.M(obj)
      v = obj.rcObj.M;
    end
    function v = get.metaNUse(obj)
      v = obj.rcObj.metaNUse;
    end    
  end      
  
  methods
    function obj = CPRVizTrackDiags(lObj,hFig)
      obj.hFig = hFig;
      
      assert(isa(lObj,'Labeler'));      
      obj.lObj = lObj;
      obj.tObj = lObj.tracker;
      obj.rcObj = lObj.tracker.trnResRC;      
    end
    function delete(obj)
      obj.cleanupHViz();
      delete(obj.hFig);
      obj.hFig = [];
    end
    function init(obj)
      obj.isinit = true;
      
      obj.gdata = guidata(obj.hFig);

      obj.iRep = 1;
      obj.t = 1;
      obj.u = 1;
      
      assert(~obj.lObj.isMultiView,'Currently unsupported for multiview projs.');
      
      obj.cleanupHViz();
      obj.hViz = cell(obj.M,obj.metaNUse);
      
      obj.isinit = false;
    end
    function fireSetObs(obj)
      mc = meta.class.fromName('CPRVizTrackDiags');
      props = mc.PropertyList;
      props = props([props.SetObservable]);
      for i=1:numel(props)
        p = props(i).Name;
        obj.(p) = obj.(p);
      end
    end
    function cleanupHViz(obj)
      if ~isempty(obj.hViz)
        for i=1:numel(obj.hViz)
          deleteValidHandles(obj.hViz{i});
        end
      end
      obj.hViz = [];
    end
  end
  methods    
    function [ipts,ftrtype] = getLandmarksUsed(obj)
      % f: [nMinor x M x nUse]
      rc = obj.rcObj;
      [ipts,ftrtype] = rc.getLandmarksUsed(obj.t);
    end
    function [fUse,xsUse,xsLbl] = vizUpdate(obj)      
      % fuse: [MxnUse] feature indices used
      % xsUse: [MxnUse] cell array, feature definitions (row of ftrSpec.xs)
      % xsLbl: [1xncol] labels for row vecs in xsUse
      
      rc = obj.rcObj;
      fUse = squeeze(rc.ftrsUse(obj.t,obj.u,:,:)); % [MxnUse]
      fspec = rc.ftrSpecs{obj.t};
      ax = obj.lObj.gdata.axes_curr;
      
      trkPFull = obj.tObj.getTrackResFull(obj.lObj.currMovie,obj.lObj.currFrame);
      % [nptstrk x d x nRep x (T+1)] 
      trkPFull = trkPFull(:,:,obj.iRep,obj.t); % [nptstrkx2]
      nptstrk = size(trkPFull,1);
      nview = 1;
      xLM = reshape(trkPFull(:,1),[1 nptstrk nview]);
      yLM = reshape(trkPFull(:,2),[1 nptstrk nview]);
      % Compute2LM
      
      % viz2LM
      clrs = lines(obj.M);
      xsUse = cell(obj.M,obj.metaNUse);
      for iFern=1:obj.M
        for iUse=1:obj.metaNUse
          switch fspec.type
            case '1lm'
            case '2lm'
              [xF,yF,chan,iview,info] = Features.compute2LM(fspec.xs,xLM,yLM);
              xsLbl = {'lm1' 'lm2' 'rfac' 'theta' 'ctrfac' 'chan' 'view'}; % TODO: hardcoded, should get it from Features or similar

              iN = 1;
              iF = fUse(iFern,iUse);
              hPlot = Features.visualize2LM(ax,xF,yF,iview,info,iN,iF,...
                clrs(iFern,:),'hPlot',obj.hViz{iFern,iUse});
              obj.hViz{iFern,iUse} = hPlot;
              
              xsUse{iFern,iUse} = fspec.xs(iF,:);              
            case '2lmdiff'
            otherwise
              assert(false);
          end        
        end
      end
    end
    function vizHiliteFernSet(obj,iFern)      
      assert(1<=iFern && iFern<=obj.M);
      for iF=1:obj.M
        for iUse=1:obj.metaNUse
          hs = obj.hViz{iF,iUse};
          if iF==iFern
            [hs.LineWidth] = deal(2);
            [hs.Visible] = deal('on');
          else
            [hs.LineWidth] = deal(1);
            [hs.Visible] = deal('off');
          end
        end
      end
    end
    function vizHiliteFernClear(obj)
      for iF=1:obj.M
        for iUse=1:obj.metaNUse
          hs = obj.hViz{iF,iUse};
          [hs.LineWidth] = deal(1);
          [hs.Visible] = deal('on');
        end
      end
    end
    function vizHide(obj)
      hV = obj.hViz;
      for i=1:numel(hV)
        [hV{i}.Visible] = deal('off');
      end
    end
    function vizShow(obj)
      hV = obj.hViz;
      for i=1:numel(hV)
        [hV{i}.Visible] = deal('on');
      end      
    end
  end  
end
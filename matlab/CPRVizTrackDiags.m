classdef CPRVizTrackDiags < handle
  properties
    isinit % true during initialization
    
    hFig % CPRVizTrackDiagsGUI
    gdata % handles
    
    lObj % Labeler obj
    tObj % Tracker obj
    rcObj % RegressorCascade obj
    
    hLM % [npts] plot handles for landmarks for current replicate
    hLMTxt % [npts]
    hViz % [MxnUse] cell array of handles for visualization     
  end
  properties (SetObservable)    
    iRep % replicate index
    t % major iter
    u % minor iter
  end
  properties (SetObservable, SetAccess=private)    
    % These props govern what is shown for 'Feature Details'
    iFernHilite % scalar, in 0..M. If 0, then no fern hilighted, all feature details shown. Otherwise, show details only for given fern.
    vizShow % scalar logical
  end
  properties (Dependent,SetAccess=private)
    nPts % number of label points
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
    function v = get.nPts(obj)
      v = obj.lObj.nLabelPoints;
    end
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
      assert(isa(obj.tObj,'CPRLabelTracker'));
      obj.rcObj = lObj.tracker.trnResRC;
    end
    function delete(obj)
      deleteValidGraphicsHandles(obj.hLM);
      deleteValidGraphicsHandles(obj.hLMTxt);
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

      obj.iFernHilite = 0;
      obj.vizShow = 1;

      obj.vizLMInit();
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
          deleteValidGraphicsHandles(obj.hViz{i});
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
      % Update Feature Details (.hViz) and Landmarks (.hLM, .hLMTxt) --
      % locations only, not visibility
      %
      % fuse: [MxnUse] feature indices used
      % xsUse: [MxnUse] cell array, feature definitions (row of ftrSpec.xs)
      % xsLbl: [1xncol] labels for row vecs in xsUse
      
      assert(~obj.lObj.gtIsGTMode,'Unsupported for GT mode.');
      
      rc = obj.rcObj;
      fUse = squeeze(rc.ftrsUse(obj.t,obj.u,:,:)); % [MxnUse]
      fspec = rc.ftrSpecs{obj.t};
      xsLbl = Features.TYPE2XSCOLLBLS(fspec.type);
      if istable(fspec.xs)
        assert(isempty(xsLbl));
        xsLbl = fspec.xs.Properties.VariableNames;
      end
      xsUse = arrayfun(@(iF)fspec.xs(iF,:),fUse,'uni',0);
      
      mIdx = obj.lObj.currMovIdx;
      frm = obj.lObj.currFrame; 
      trkPFull = obj.tObj.getTrackResFullCurrTgt(mIdx,frm);
      
      if isequal(trkPFull,[])
        % no tracking avail for this iMov/frm
        for iFern=1:obj.M
          for iUse=1:obj.metaNUse
            h = obj.hViz{iFern,iUse};
            set(h,'XData',nan,'YData',nan); % Works for empty h
          end
        end
        set(obj.hLM,'XData',nan,'YData',nan);
        set(obj.hLMTxt,'Position',[nan nan 1]);
      else
        % trkPFull is [nptstrk x d x nRep x (T+1)] 
        
        % Get xLM/yLM, landmark positions at this mov/frm/replicate/majoriter
        trkPFull = trkPFull(:,:,obj.iRep,obj.t); % [nptstrkx2]
        nptstrk = size(trkPFull,1);
        nview = 1;
        xLM = reshape(trkPFull(:,1),[1 nptstrk nview]);
        yLM = reshape(trkPFull(:,2),[1 nptstrk nview]);

        ax = obj.lObj.gdata.axes_curr;      
        clrs = rgbbrighten(lines(obj.M),0.5);
        for iFern=1:obj.M
          for iUse=1:obj.metaNUse
            iFuse = fUse(iFern,iUse);
            switch fspec.type
              case 'single landmark'                
                [xF,yF,iview,info] = Features.compute1LM(fspec.xs(iFuse,:),xLM,yLM);
                hPlot = Features.visualize1LM(ax,xF,yF,iview,info,1,1,...
                  clrs(iFern,:),'hPlot',obj.hViz{iFern,iUse});
              case '2lm'
                [xF,yF,~,iview,info] = Features.compute2LM(fspec.xs(iFuse,:),xLM,yLM);
                hPlot = Features.visualize2LM(ax,xF,yF,iview,info,1,1,...
                  clrs(iFern,:),'hPlot',obj.hViz{iFern,iUse});
              case 'two landmark elliptical'
                [xF,yF,~,iview,info] = Features.compute2LMelliptical(fspec.xs(iFuse,:),xLM,yLM);
                hPlot = Features.visualize2LMelliptical(ax,xF,yF,iview,info,1,1,...
                  clrs(iFern,:),'hPlot',obj.hViz{iFern,iUse});                
              case '2lmdiff'
                assert(false,'Currently unsupported');
              otherwise
                assert(false);
            end   
            obj.hViz{iFern,iUse} = hPlot;
          end
        end

        obj.vizLMUpdate([xLM(:) yLM(:)]);
      end
    end
    function vizLMInit(obj)
      deleteValidGraphicsHandles(obj.hLM);
      deleteValidGraphicsHandles(obj.hLMTxt);
      obj.hLM = gobjects(obj.nPts,1);
      obj.hLMTxt = gobjects(obj.nPts,1);
      
      ax = obj.lObj.gdata.axes_curr;
      plotIfo = obj.lObj.labelPointsPlotInfo;
      for i = 1:obj.nPts
        obj.hLM(i) = plot(ax,nan,nan,'^',...
          'MarkerSize',plotIfo.MarkerSize,...
          'LineWidth',plotIfo.LineWidth,...
          'Color',plotIfo.Colors(i,:),...
          'MarkerFaceColor',plotIfo.Colors(i,:),...
          'PickableParts','none');
        obj.hLMTxt(i) = text(nan,nan,num2str(i),'Parent',ax,...
          'Color',plotIfo.Colors(i,:),'PickableParts','none');
      end
    end
    function vizLMUpdate(obj,xyLM)
      npts = obj.nPts;
      szassert(xyLM,[npts 2]);
      LabelCore.setPtsCoordsStc(xyLM,obj.hLM,obj.hLMTxt,...
        obj.lObj.labelPointsPlotInfo.LblOffset);
    end
  end
  methods % Feature Detail Visibility
    function vizDetailUpdate(obj)
      % update .hViz based on .iFernHilite, .vizShow
      
      for iF=1:obj.M
        if obj.vizShow
          tfShowFern = obj.iFernHilite==0 || obj.iFernHilite==iF;
        else
          tfShowFern = false;
        end
        showFernOnOff = onIff(tfShowFern); 
        lwidth = 1;
%         if iF==obj.iFernHilite
%           lwidth = 2;
%         else
%           lwidth = 1;
%         end
        for iUse=1:obj.metaNUse
          hs = obj.hViz{iF,iUse};
          [hs.Visible] = deal(showFernOnOff);
          [hs.LineWidth] = deal(lwidth);
        end
      end      
    end      
    function vizHiliteFernSet(obj,iFern)
      assert(0<=iFern && iFern<=obj.M);
      obj.iFernHilite = iFern;
      obj.vizDetailUpdate();
    end
    function vizDetailHide(obj)
      obj.vizShow = false;
      obj.vizDetailUpdate();
    end
    function vizDetailShow(obj)
      obj.vizShow = true;
      obj.vizDetailUpdate();
    end
  end
end
classdef TrackingVisualizer < handle
  
  % TrackingVisualizers know how to plot/show tracking results on an axes
  % (not owned by itself). They know how to show things and they own the 
  % relevant lines/graphics handles but that's it. Theoretically you can 
  % create/delete them at will to add/rm tracking overlays on top of your
  % images/movies.

  % LabelTracker Property forwarding notes 20181211 LabelTracker contains 
  % stateful props (.hideViz, .hideVizTxt, .showVizReplicates) that are 
  % conceptually forwarding props to TrackingVisualizer. They are there 
  % because i) LT currently presents a single public interface to clients 
  % for tracking (including SetObservability); and ii) LT handles 
  % serialization/reload of these props. This seems fine for now.
  %
  % The pattern followed here for these props is
  % - stateful, SetObservable prop in LT
  % - stateful prop in TrkVizer. Note a lObj has multiple TVs.
  % - set methods set stateful prop and forward to trkVizer, which performs
  % action and sets their stateful props
  % - no getters, just get the prop
  % - get/loadSaveToken set the stateful prop and forward to trkVizer

  properties 
    lObj % Included only to access the current raw image. Ideally used as little as possible

    hIms % [nview] image handles. Owned by Labeler
    hAxs % [nview] axes handles. Owned by Labeler

    ipt2vw % [npts], like Labeler/labeledposIPt2View
    ptClrs % [nptsx3], like Labeler/labeledposIPt2View.
    tfocc % [npts], logical flag for occluded prediction
   
    txtOffPx % scalar, px offset for landmark text labels 

    tfHideViz % scalar, true if tracking res hidden
    tfHideTxt % scalar, if true then hide text even if tfHideViz is false

    % besides colors, txtOffPx, the the show/hide state, other cosmetic 
    % state is stored just in the various graphics handles.
    %
    % Note that at least one visibility flag must be stored outside the
    % handles themselves, since text and markers+text can be independently 
    % shown/hidden
        
    handleTagPfix % char, prefix for handle tags

    hXYPrdRed; % [npts] plot handles for 'reduced' tracking results, current frame and target
    hXYPrdRedOther; % [npts] plot handles for 'reduced' tracking results, current frame, non-current-target
    hXYPrdRedTxt; % [nPts] handle vec, text labels for hXYPrdRed
    
    hXYPrdRedMrkr % char. .Marker for hXYPrdRed. Needs to be stored here due b/c 
            % TrackingVisualizer displays occluded tracking results with a
            % different marker.
  end
  properties (Constant)
    SAVEPROPS = {'ipt2vw' 'ptClrs' 'txtOffPx' 'tfHideViz' 'tfHideTxt' ...
      'handleTagPfix'};
    LINE_PROPS_COSMETIC_SAVE = {'Color' 'LineWidth' 'Marker' ...
      'MarkerEdgeColor' 'MarkerFaceColor' 'MarkerSize'};
    TEXT_PROPS_COSMETIC_SAVE = {'FontSize' 'FontName' 'FontWeight' 'FontAngle'};
    
    OCCLUDED_MARKER = 'o';
  end
  properties (Dependent)
    nPts
  end
  methods
    function v = get.nPts(obj)
      v = numel(obj.ipt2vw);
    end    
  end
  
  methods
    function deleteGfxHandles(obj)
      if ~isstruct(obj.hXYPrdRed) % guard against serialized TVs which have PV structs in .hXYPrdRed
        deleteValidGraphicsHandles(obj.hXYPrdRed);
        obj.hXYPrdRed = [];
      end
      deleteValidGraphicsHandles(obj.hXYPrdRedOther);
      obj.hXYPrdRedOther = [];
      deleteValidGraphicsHandles(obj.hXYPrdRedTxt);
      obj.hXYPrdRedTxt = [];
    end
    function vizInit(obj,varargin)
      % Sets .hXYPrdRed, .hXYPrdRedOther, .hXYPrdRedTxt.
      % 
      % See "Construction/Init notes" below      

      postload = myparse(varargin,...
        'postload',false... % see Construction/Init notes
        );      
      
      obj.deleteGfxHandles();
      
      pppi = obj.lObj.predPointsPlotInfo;

      npts = numel(obj.ipt2vw);
      if postload
        ptclrs = obj.ptClrs;
      else
        ptclrs = obj.lObj.PredictPointColors;
        obj.ptClrs = ptclrs;
        obj.txtOffPx = pppi.TextOffset;
      end
      szassert(ptclrs,[npts 3]);
      
      obj.tfocc = false(npts,1);

      % init .xyVizPlotArgs*
      markerPVs = pppi.MarkerProps;
      textPVs = pppi.TextProps;
      markerPVs.PickableParts = 'none';
      textPVs.PickableParts = 'none';
      markerPVs = struct2paramscell(markerPVs);
      textPVs = struct2paramscell(textPVs);
      %markerPVsNonTarget = markerPVs; % TODO: customize
      
      if postload
        % We init first with markerPVs/textPVs, then set saved custom PVs
        hXYPrdRed0 = obj.hXYPrdRed;
        hXYPrdRedTxt0 = obj.hXYPrdRedTxt;
      end
      
      npts = obj.nPts;
      ax = obj.hAxs;
      arrayfun(@(x)hold(x,'on'),ax);
      ipt2View = obj.ipt2vw;
      ipt2set = obj.lObj.labeledposIPt2Set;
      hTmp = gobjects(npts,1);
      hTmpOther = gobjects(npts,1);
      hTxt = gobjects(npts,1);
      pfix = obj.handleTagPfix;
      for iPt = 1:npts
        clr = ptclrs(iPt,:);
        iVw = ipt2View(iPt);
        ptset = ipt2set(iPt);
        hTmp(iPt) = plot(ax(iVw),nan,nan,markerPVs{:},...
          'Color',clr,...
          'Tag',sprintf('%s_XYPrdRed_%d',pfix,iPt));
        hTmpOther(iPt) = plot(ax(iVw),nan,nan,markerPVs{:},...
          'Color',clr,...
          'Tag',sprintf('%s_XYPrdRedOther_%d',pfix,iPt));
        hTxt(iPt) = text(nan,nan,num2str(ptset),'Parent',ax(iVw),...
          'Color',clr,textPVs{:},...
          'Tag',sprintf('%s_PrdRedTxt_%d',pfix,iPt));
      end
      obj.hXYPrdRed = hTmp;
      obj.hXYPrdRedOther = hTmpOther;
      obj.hXYPrdRedTxt = hTxt;
            
      if postload
        if isstruct(hXYPrdRed0)
          if numel(hXYPrdRed0)==numel(hTmp)
            arrayfun(@(x,y)set(x,y),hTmp,hXYPrdRed0);
          else
            warningNoTrace('.hXYPrdRed: Number of saved prop-val structs does not match number of line handles.');
          end
        end
        if isstruct(hXYPrdRedTxt0)
          if numel(hXYPrdRedTxt0)==numel(hTxt)
            arrayfun(@(x,y)set(x,y),hTxt,hXYPrdRedTxt0);
          else
            warningNoTrace('.hXYPrdRedTxt: Number of saved prop-val structs does not match number of line handles.');
          end
        end
      end
      
      obj.hXYPrdRedMrkr = get(hTmp(1),'Marker');

      % default textPVs do not respect .tfHideViz/.tfHideTxt
      obj.updateHideVizHideText(); 
      
      obj.vizInitHook();
    end
    function vizInitHook(obj)
      % overload me
    end
    function setHideViz(obj,tf)
      obj.tfHideViz = tf;
      obj.updateHideVizHideText();
    end
    function setHideTextLbls(obj,tf)
      obj.tfHideTxt = tf;
      obj.updateHideVizHideText();
    end
    function updateHideVizHideText(obj)
      onoffViz = onIff(~obj.tfHideViz);
      [obj.hXYPrdRed.Visible] = deal(onoffViz);
      [obj.hXYPrdRedOther.Visible] = deal(onoffViz);
      onoffTxt = onIff(~obj.tfHideViz && ~obj.tfHideTxt);
      [obj.hXYPrdRedTxt.Visible] = deal(onoffTxt);
    end
    function updateLandmarkColors(obj,ptsClrs)
      npts = obj.nPts;
      szassert(ptsClrs,[npts 3]);
      for iPt=1:npts
        clr = ptsClrs(iPt,:);
        set(obj.hXYPrdRed(iPt),'Color',clr);
        set(obj.hXYPrdRedOther(iPt),'Color',clr);
        set(obj.hXYPrdRedTxt(iPt),'Color',clr);
      end
      obj.ptClrs = ptsClrs;
    end
    function updateTrackRes(obj,xy,tfocc)
      %
      % xy: [npts x 2]
            
      npts = obj.nPts;
      h = obj.hXYPrdRed;
      hTxt = obj.hXYPrdRedTxt;
      dx = obj.txtOffPx;
      xyoff = xy+dx;
      for iPt=1:npts
        set(h(iPt),'XData',xy(iPt,1),'YData',xy(iPt,2));
        set(hTxt(iPt),'Position',[xyoff(iPt,:) 0]);
      end
      
      if nargin==3
        assert(numel(tfocc)==numel(h));
        set(h(tfocc),'Marker',obj.OCCLUDED_MARKER);
        set(h(~tfocc),'Marker',obj.hXYPrdRedMrkr);
        obj.tfocc = tfocc;
      end
    end
    function setMarkerCosmetics(obj,pvargs)
      if isstruct(pvargs)
        arrayfun(@(x)set(x,pvargs),obj.hXYPrdRed);
      else
        arrayfun(@(x)set(x,pvargs{:}),obj.hXYPrdRed);
      end
      % TODO: If some markers are currently occluded, this will 'overwrite'
      % them with normal makrers. Browse to a new frame will refresh.
      obj.hXYPrdRedMrkr = get(obj.hXYPrdRed(1),'Marker');
    end
    function setTextCosmetics(obj,pvargs)
      if isstruct(pvargs)
        arrayfun(@(x)set(x,pvargs),obj.hXYPrdRedTxt);
      else        
        arrayfun(@(x)set(x,pvargs{:}),obj.hXYPrdRedTxt);
      end
    end
    function setTextOffset(obj,dx)
      obj.txtOffPx = dx;
            
      npts = obj.nPts;
      h = obj.hXYPrdRed;
      hTxt = obj.hXYPrdRedTxt;
      x = get(h,'XData');
      y = get(h,'YData');
      xy = [cell2mat(x(:)) cell2mat(y(:))];
      %szassert(xy,[npts 2]);      
      xyoff = xy+dx;
      
      for iPt=1:npts
        set(hTxt(iPt),'Position',[xyoff(iPt,:) 0]);
      end
    end
  end
  
  methods 
    % Construction/Init notes 
    %
    % 1. Call the constructor normally, then vizInit();
    %   - This initializes cosmetics from labeler.predPointsPlotInfo
    %   - This is the codepath used for LabelTrackers. LabelTracker TVs
    %   are not serialized. New/fresh ones are created and cosmetics are
    %   initted from labeler.predPointsPlotInfo.
    % 2. From serialized. Call constructor with no args, then postLoadInit()
    %   - SaveObj restores various cosmetic state, including PV props in
    %   .hXYPrdRed and .hXYPrdRedTxt
    %   - PostLoadInit->vizInit sets up cosmetic state on handles
    %
    % Save/load strategy. 
    %
    % In saveobj we record the cosmetics used for a TrackingVisualizer for 
    % the .hXYPrdRed line handles by doing a get and saving the resulting 
    % PVs in .hXYPrdRed; similarly for .hXYPrdRedTxt.
    %
    % Loadobj keeps these PVs in .hXYPrdRed and .hxYPrdRedTxt. At 
    % postLoadInit->vizInit('postload',true) time, the PVs are re-set on 
    % the .hXYPrdRed line handles. In this way, serialized TVs can keep
    % arbitrary customized cosmetics.
    
    function obj = TrackingVisualizer(lObj,handleTagPfix)
      obj.tfHideTxt = false;
      obj.tfHideViz = false;            

      if nargin==0
        return;
      end
      
      obj.lObj = lObj;
      gd = lObj.gdata;
      obj.hAxs = gd.axes_all;
      obj.hIms = gd.images_all;
      obj.ipt2vw = lObj.labeledposIPt2View;    
      
      obj.handleTagPfix = handleTagPfix;
    end
    function postLoadInit(obj,lObj)
      obj.lObj = lObj;
      gd = lObj.gdata;
      obj.hAxs = gd.axes_all;
      obj.hIms = gd.images_all;

      assert(isequal(obj.ipt2vw,lObj.labeledposIPt2View));
      
      obj.vizInit('postload',true);
    end
    function delete(obj)
      obj.deleteGfxHandles();
    end
    
    function s = saveobj(obj)
      s = struct();
      for p=TrackingVisualizer.SAVEPROPS,p=p{1}; %#ok<FXSET>
        s.(p) = obj.(p);
      end
      
      lineprops = obj.LINE_PROPS_COSMETIC_SAVE;
      vals = get(obj.hXYPrdRed,lineprops); % [nhandle x nprops]
      s.hXYPrdRed = cell2struct(vals,lineprops,2);
      % save assuming none are occluded
      [s.hXYPrdRed.Marker] = deal(obj.hXYPrdRedMrkr); 
      
      textprops = obj.TEXT_PROPS_COSMETIC_SAVE;
      vals = get(obj.hXYPrdRedTxt,textprops); % [nhandle x nprops]
      s.hXYPrdRedTxt = cell2struct(vals,textprops,2);
    end
  end
  methods (Static)
    function b = loadobj(a)
      if isstruct(a)
        b = TrackingVisualizer();
        for p=TrackingVisualizer.SAVEPROPS,p=p{1}; %#ok<FXSET>
          b.(p) = a.(p);
        end
        b.hXYPrdRed = a.hXYPrdRed;
        if isfield(a,'hXYPrdRedTxt')
          b.hXYPrdRedTxt = a.hXYPrdRedTxt;
        end
      else
        b = a;
      end
    end
  end
end
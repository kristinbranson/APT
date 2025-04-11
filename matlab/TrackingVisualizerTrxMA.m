classdef TrackingVisualizerTrxMA < handle
  % shows a trx/centroid marker, text label, trajectory traces
  
  % Conceptually similar to TrackingVisualizerTrx but as prototyping MA 
  % wanted to dev without worrying about SA-Trx regressions
  
  properties
    lObj
    
    hTraj;                    % nTrx x 1 vector of line handles
    hTrx;                     % nTrx x 1 vector of line handles
    hTrxTxt;                  % nTrx x 1 vector of text handles    
    showTrxPreNFrm = 15;      % number of preceding frames to show in traj
    showTrxPostNFrm = 5;      % number of following frames to show in traj
    
    currTrx;                  % index into hTrx for current/selected trx
                              % 0 <-> none currently 
    nTrxLive                  % Current number of live trx
    
    clrsTrx;                  % nTrx x 3 vector of colors; applied to hTrx, hTraj, etc
                              % init'd at init() time
    clrTrxCurrent;            % 1x3 color for current trx
    
    trxClickable = true;
    trxSelectCbk;             % cbk with sig trxSelectCbk(iTrx); called when 
                              % trxClickable=true and on trx BDF
                              
    tfHideViz = false;
    showOnlyPrimary = false;  % if true, only show currTrx etc    
  end
  properties (Dependent)
    nTrx
  end
  
  methods 
    function v = get.nTrx(obj)
      v = numel(obj.hTrx);
    end
  end
      
  methods
    
    function obj = TrackingVisualizerTrxMA(labeler)
      obj.lObj = labeler;
    end
    function delete(obj)
      obj.deleteGfxHandles();
    end
    function deleteGfxHandles(obj)
      deleteValidGraphicsHandles(obj.hTraj);
      obj.hTraj = [];
      deleteValidGraphicsHandles(obj.hTrx);
      obj.hTrx = [];
      deleteValidGraphicsHandles(obj.hTrxTxt);
      obj.hTrxTxt = [];
    end
    
    function init(obj,trxSelCbk,nTrx)
      % create gfx handles
      %
      % click2nav: either true, false, or arbitrary callback called for trx
      %           bdf. Sig: cbk(iTrx)
      
      obj.deleteGfxHandles();
      
      lObj = obj.lObj; %#ok<*PROPLC>
              
      deleteValidGraphicsHandles(obj.hTraj);
      deleteValidGraphicsHandles(obj.hTrx);
      deleteValidGraphicsHandles(obj.hTrxTxt);
      obj.hTraj = matlab.graphics.primitive.Line.empty(0,1);
      obj.hTrx = matlab.graphics.primitive.Line.empty(0,1);
      obj.hTrxTxt = matlab.graphics.primitive.Text.empty(0,1);
      
      obj.currTrx = 0;
      obj.nTrxLive = 0;
      
      assert(isa(trxSelCbk,'function_handle'));
      obj.trxSelectCbk = trxSelCbk;
      bdf = @(src,evt)obj.bdfTrx(src,evt);
      obj.trxClickable = true;      
      clrsT = obj.setColors(nTrx);
      
      ax = lObj.gdata.axes_curr;
      pref = lObj.projPrefs.Trx;
      for i = 1:nTrx 
        clr = clrsT(i,:);
        obj.hTraj(i,1) = line(...
          'parent',ax,...
          'xdata',nan, ...
          'ydata',nan, ...
          'color',clr,...
          'linestyle',pref.TrajLineStyle, ...
          'linewidth',pref.TrajLineWidth, ...
          'HitTest','off',...
          'Tag',sprintf('Labeler_Traj_%d',i),...
          'PickableParts','none');

        obj.hTrx(i,1) = plot(ax,nan,nan,pref.TrxMarker);               
        set(obj.hTrx(i,1),...
          'Color',clr,...
          'MarkerSize',pref.TrxMarkerSize,...
          'LineWidth',pref.TrxLineWidth,...
          'Tag',sprintf('Labeler_Trx_%d',i),...
          'ButtonDownFcn',bdf,...
          'UserData',i,...
          'PickableParts','all',...
          'HitTest','on');
        
        obj.hTrxTxt(i,1) = text(nan,nan,num2str(i),'Parent',ax,...
          'Color',clr,...
          'Fontsize',pref.TrxIDLblFontSize,...
          'Fontweight',pref.TrxIDLblFontWeight,...
          'PickableParts','none',...
          'Tag',sprintf('Labeler_TrxTxt_%d',i));
      end
    end
    
    function clrsT = setColors(obj,nTrx)
      % init/set .clrsTrx, .clrTrxCurrent from lObj.projPrefs.Trx
      
      if nargin<2
        nTrx = obj.nTrx;
      end
      
      prefsTrx = obj.lObj.projPrefs.Trx;
      ptcolor = prefsTrx.TrajColor;
      if ischar(ptcolor)
        % should not get hit; currently not using prefs.Trx.TrajColormapName
        clrsT = eval(ptcolor,nTrx);
      else
        assert(isnumeric(ptcolor));
        nptc = size(ptcolor,1);
        nreps = ceil(nTrx/nptc);
        clrsT = repmat(ptcolor,nreps,1);
        clrsT = clrsT(1:nTrx,:);
      end
      obj.clrsTrx = clrsT;
      obj.clrTrxCurrent = prefsTrx.TrajColorCurrent;
    end
    
    function updateColors(obj)
      clrsT = obj.setColors();
      for iTrx = 1:obj.nTrx 
        if iTrx==obj.currTrx
          clr = obj.clrTrxCurrent;
        else
          clr = clrsT(iTrx,:);
        end
        obj.hTraj(iTrx).Color = clr;
        obj.hTrx(iTrx).Color = clr;
        obj.hTrxTxt(iTrx).Color = clr;
      end
    end
    
    function bdfTrx(obj,src,~)
      iTrx = src.UserData;
      obj.updatePrimaryTrx(iTrx);
      obj.trxSelectCbk(iTrx);
    end
    
    function updatePrimaryTrx(obj,iTrxPrimary)
      % use iTrxPrimary==0 for "no current"
      obj.currTrx = iTrxPrimary;
      obj.updateColors();
      if obj.showOnlyPrimary
        tfShowTrx = obj.computeTfShowTrx();
        obj.setShowTrx(tfShowTrx);        
      end
    end
    
    function updateLiveTrx(obj,trxLive,frm,tfUpdateIDs)
      % Set/update positions of all live trx/trajs
      %
      % trxAll: [ntrxlive] trx struct array; ntrxlivemust be <= obj.nTrx
      % frm: current frame
      % tfUpdateIDs: if true, trxAll must have IDs set and hTrxTxt are
      %   updated per these IDs.
            
      nPre = obj.showTrxPreNFrm;
      nPst = obj.showTrxPostNFrm;
      lObj = obj.lObj;
      pref = lObj.projPrefs.Trx;
      dx = pref.TrxIDLblOffset;

      ntrxlive = numel(trxLive);
      assert(ntrxlive<=obj.nTrx);
      obj.nTrxLive = ntrxlive;
      
      tfShowTrx = obj.computeTfShowTrx();
      obj.setShowTrx(tfShowTrx);

%     20221201 we update all positions here even for visible='off' gfx
%     handles. Otherwise, toggling visibility would reveal incorrect
%     positions. Notes:
%     * The performance hit does not seem significant so far. This update
%       is not a bottleneck eg when Playing a tracked movie.
%     * Alternative would be to lazy-update on any toggling of visibility/
%       primariness. 
%
%       iShowTrx = find(tfShowTrx);      
%       for iTrx = iShowTrx(:)'
      for iTrx=1:ntrxlive
        
        trxCurr = trxLive(iTrx);
        t0 = trxCurr.firstframe;
        t1 = trxCurr.endframe;      
        if t0<=frm && frm<=t1 && ~isempty(trxCurr.x)
          idx = frm+trxCurr.off;
          xTrx = trxCurr.x(idx);
          yTrx = trxCurr.y(idx);
        else
          xTrx = nan;
          yTrx = nan;
        end
        set(obj.hTrx(iTrx),'XData',xTrx,'YData',yTrx);
        
        tTraj = max(frm-nPre,t0):min(frm+nPst,t1); % could be empty array
        iTraj = tTraj + trxCurr.off;
        if ~isempty(trxCurr.x)
          xTraj = trxCurr.x(iTraj);
          yTraj = trxCurr.y(iTraj);
        else
          xTraj = nan;
          yTraj = nan;
        end
       
        set(obj.hTraj(iTrx),'XData',xTraj,'YData',yTraj);
        %set(obj.hTrx(iTrx),'Color',color);
        
        if lObj.showTrxIDLbl
          set(obj.hTrxTxt(iTrx),'Position',[xTrx+dx yTrx+dx 1]);
          if tfUpdateIDs
            idstr = num2str(trxCurr.id);
            set(obj.hTrxTxt(iTrx),'String',idstr);
          end
        end        
      end
    end
    
    function setShowTrx(obj,tfShowTrx)
      % relies on obj.lObj.showTrxIDLbl
      
      set(obj.hTraj(tfShowTrx),'Visible','on');
      set(obj.hTraj(~tfShowTrx),'Visible','off');
      set(obj.hTrx(tfShowTrx),'Visible','on');
      set(obj.hTrx(~tfShowTrx),'Visible','off');
      set(obj.hTraj(tfShowTrx),'HitTest','on');
      set(obj.hTraj(~tfShowTrx),'HitTest','off');
      set(obj.hTrx(tfShowTrx),'HitTest','on');
      set(obj.hTrx(~tfShowTrx),'HitTest','off');
      
      if obj.lObj.showTrxIDLbl
        set(obj.hTrxTxt(tfShowTrx),'Visible','on');
        set(obj.hTrxTxt(~tfShowTrx),'Visible','off');
        set(obj.hTrxTxt(tfShowTrx),'HitTest','on');
        set(obj.hTrxTxt(~tfShowTrx),'HitTest','off');
      else
        set(obj.hTrxTxt,'Visible','off');
      end
    end
    
    function tfShowTrx = computeTfShowTrx(obj)
      ntrxlive = obj.nTrxLive;
      tfShowTrx = false(obj.nTrx,1);
      if obj.tfHideViz
        % none
      elseif obj.showOnlyPrimary
        cTrx = obj.currTrx;
        if cTrx>0 && cTrx<=ntrxlive
          tfShowTrx(cTrx) = true;
        end
      else
        tfShowTrx(1:ntrxlive) = true;
      end
    end
    
    function updateShowHideAll(obj)
      % sets visibility of .hTraj, .hTrx, .hTrxTxt based on .tfHideViz, 
      % .showOnlyPrimary etc 
      tfShow = obj.computeTfShowTrx();
      obj.setShowTrx(tfShow);      
    end    
    
    function setHideViz(obj,tf)
      obj.tfHideViz = tf;
      obj.updateShowHideAll();
    end
    
    function setAllShowHide(obj,tfHideOverall,tfShowCurrTgtOnly)
      obj.tfHideViz = tfHideOverall;
      obj.showOnlyPrimary = tfShowCurrTgtOnly;
      obj.updateShowHideAll();
    end

    function setShowOnlyPrimary(obj,tf)
      obj.showOnlyPrimary = tf;
      obj.updateShowHideAll();      
    end
    
    function set_hittest(obj,onoff)
      if ~isempty(obj.hTraj) % protect against rare cases uninitted obj (eg projLoad with "nomovie")
        [obj.hTraj.HitTest] = deal(onoff);
        [obj.hTrx.HitTest] = deal(onoff);
        [obj.hTrxTxt.HitTest] = deal(onoff);
      end      
    end
    
    function hittest_off_all(obj)
      obj.set_hittest('off');
    end
    function hittest_on_all(obj)
      obj.set_hittest('on');
    end
  
%   Currently trx txtlbl viz controlled by lObj.showTrxIDLbl    
%     function setHideTextLbls(obj,tfshow)
%     end
    
  end
  
end
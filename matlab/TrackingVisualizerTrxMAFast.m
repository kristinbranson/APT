classdef TrackingVisualizerTrxMAFast < handle
  % shows a trx/centroid marker, text label, trajectory traces
  
  % Conceptually similar to TrackingVisualizerTrx but as prototyping MA 
  % wanted to dev without worrying about SA-Trx regressions
  
  properties
    lObj
    
    hTraj;                    % 1 x 1 vector of line handles
    hTrx;                     % 1 x 1 vector of line handles
    hTrxTxt;                  % 1 x nTrx vector of text handles    
    hTrajPrim;                 % 1 x 1 vector of line handles for prim
    hTrxPrim;                 % 1 x 1 vector of line handles for prim
    hTrxTxtPrim;                 % 1 x 1 vector of text handles for prim
    showTrxPreNFrm = 15;      % number of preceding frames to show in traj
    showTrxPostNFrm = 5;      % number of following frames to show in traj
    xtrj = [];
    ytrj = [];
    ids = [];
    
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
    nTrx
  end

  methods
    
    function obj = TrackingVisualizerTrxMAFast(labeler)
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
      
      obj.hTraj = line(...
        'parent',ax,...
        'xdata',nan, ...
        'ydata',nan, ...
        'color',clrsT,...
        'linestyle',pref.TrajLineStyle, ...
        'linewidth',pref.TrajLineWidth, ...
        'HitTest','off',...
        'Tag',sprintf('Labeler_Traj'),...
        'PickableParts','none');
      obj.hTrajPrim = line(...
        'parent',ax,...
        'xdata',nan, ...
        'ydata',nan, ...
        'color',obj.clrTrxCurrent,...
        'linestyle',pref.TrajLineStyle, ...
        'linewidth',pref.TrajLineWidth, ...
        'HitTest','off',...
        'Tag',sprintf('Labeler_Traj'),...
        'PickableParts','none');

      obj.hTrx = plot(ax,nan,nan,pref.TrxMarker);
      set(obj.hTrx,...
          'Color',clrsT,...
          'MarkerSize',pref.TrxMarkerSize,...
          'LineWidth',pref.TrxLineWidth,...
          'Tag','Labeler_Trx',...
          'ButtonDownFcn',bdf,...
          ...% 'UserData',i,...
          'PickableParts','all',...
          'HitTest','on');

      obj.hTrxPrim = plot(ax,nan,nan,pref.TrxMarker);
      set(obj.hTrxPrim,...
          'Color',obj.clrTrxCurrent,...
          'MarkerSize',pref.TrxMarkerSize,...
          'LineWidth',pref.TrxLineWidth,...
          'Tag','Labeler_Trx',...
          'ButtonDownFcn',bdf,...
          ...% 'UserData',i,...
          'PickableParts','all',...
          'HitTest','off');

      textTrx = {};
      for i = 1:nTrx
        textTrx{i} = num2str(i);
      end
      obj.hTrxTxt = text(nan(1,nTrx),nan(1,nTrx),textTrx,'Parent',ax,...
        'Color',clrsT,...
        'Fontsize',pref.TrxIDLblFontSize,...
        'Fontweight',pref.TrxIDLblFontWeight,...
        'PickableParts','none',...
        'Tag',sprintf('Labeler_TrxTxt'));
      obj.hTrxTxtPrim = text(nan,nan,'','Parent',ax,...
        'Color',obj.clrTrxCurrent,...
        'Fontsize',pref.TrxIDLblFontSize,...
        'Fontweight',pref.TrxIDLblFontWeight,...
        'PickableParts','none',...
        'Tag',sprintf('Labeler_TrxTxt'));

    end
    
    function clrsT = setColors(obj,nTrx)
      % init/set .clrsTrx, .clrTrxCurrent from lObj.projPrefs.Trx
            
      prefsTrx = obj.lObj.projPrefs.Trx;
      clrsT = prefsTrx.TrajColor;
      
      obj.clrsTrx = clrsT;
      obj.clrTrxCurrent = prefsTrx.TrajColorCurrent;
    end
        
    function bdfTrx(obj,src,evt)
      % Get click coordinates
      ax = ancestor(src, 'axes');
      clickPt = ax.CurrentPoint;
      xClick = clickPt(1,1);
      yClick = clickPt(1,2);

      % Find closest point in obj.xtrj and obj.ytrj
      if ~isempty(obj.xtrj) && ~isempty(obj.ytrj)
        % Filter out NaN values
        validIdx = ~isnan(obj.xtrj) & ~isnan(obj.ytrj);
        xValid = obj.xtrj(validIdx);
        yValid = obj.ytrj(validIdx);
        idsValid = obj.ids(validIdx);

        if ~isempty(xValid)
          % Calculate distances
          dists = sqrt((xValid - xClick).^2 + (yValid - yClick).^2);
          [~, minIdx] = min(dists);
          iTrx = idsValid(minIdx);
        else
          iTrx = 1; % Default to first trajectory if no valid points
        end
      else
        iTrx = 1; % Default to first trajectory if empty
      end

      obj.updatePrimaryTrx(iTrx);
      obj.trxSelectCbk(iTrx);
    end
    
    function updatePrimaryTrx(obj,iTrxPrimary)
      % use iTrxPrimary==0 for "no current"
      obj.currTrx = iTrxPrimary;
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
      xdata = [];
      ydata = [];
      xdataPrim = [];
      ydataPrim = [];
      xtrj = [];
      ytrj = [];
      xtrjPrim = nan;
      ytrjPrim = nan;
      ids = [];
      set(obj.hTrxTxtPrim,'Position',[nan nan 1]);

      for iTrx=1:numel(trxLive)
        
        trxCurr = trxLive(iTrx);
        if iTrx == obj.currTrx
          prim = true;
        else
          prim = false;
        end

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
        xtrj(end+1) = xTrx;
        ytrj(end+1) = yTrx;
        ids(end+1) = iTrx;

        if prim
          xtrjPrim = xTrx; ytrjPrim = yTrx;
        end
        
        tTraj = max(frm-nPre,t0):min(frm+nPst,t1); % could be empty array
        iTraj = tTraj + trxCurr.off;
        if ~isempty(trxCurr.x)
          n = numel(iTraj);
          xdata(end+1:end+n) = trxCurr.x(iTraj);
          ydata(end+1:end+n) = trxCurr.y(iTraj);
          xdata(end+1) = nan;
          ydata(end+1) = nan;
          if prim
            xdataPrim = trxCurr.x(iTraj);
            ydataPrim = trxCurr.y(iTraj);
          end

          if lObj.showTrxIDLbl
            set(obj.hTrxTxt(iTrx),'Position',[xTrx+dx yTrx+dx 1]);
            idstr = num2str(trxCurr.id);
            set(obj.hTrxTxt(iTrx),'String',idstr);
            if prim
              set(obj.hTrxTxtPrim,'Position',[xTrx+dx yTrx+dx 1]);
              idstr = num2str(trxCurr.id);
              set(obj.hTrxTxtPrim,'String',idstr);
            end
          end
        else
            set(obj.hTrxTxt(iTrx),'Position',[nan nan 1]);          
        end
      end
      for iTrx = (numel(trxLive)+1):numel(obj.hTrxTxt)
        set(obj.hTrxTxt(iTrx),'Position',[nan nan 1]);
      end
      set(obj.hTrx,'xdata',xtrj,'ydata',ytrj);
      set(obj.hTrxPrim,'xdata',xtrjPrim,'ydata',ytrjPrim);
      set(obj.hTraj,'xdata',xdata,'ydata',ydata);
      set(obj.hTrajPrim,'xdata',xdataPrim,'ydata',ydataPrim);
      assert(numel(ids)==numel(xtrj));
      obj.xtrj = xtrj;
      obj.ytrj = ytrj;
      obj.ids = ids;
    end
    
    
    function setHideViz(obj,tf)
      obj.tfHideViz = tf;
      onOff = onIff(~tf);
      set(obj.hTraj,'visible',onOff);
      set(obj.hTrajPrim,'visible',onOff);
      set(obj.hTrx,'visible',onOff);
      set(obj.hTrxPrim,'visible',onOff);

      % Text labels depend on both tf and showTrxIDLbl
      if obj.lObj.showTrxIDLbl && ~tf
        set(obj.hTrxTxt,'visible','on');
        set(obj.hTrxTxtPrim,'visible','on');
      else
        set(obj.hTrxTxt,'visible','off');
        set(obj.hTrxTxtPrim,'visible','off');
      end
    end
    
    function setAllShowHide(obj,tfHideOverall,tfShowCurrTgtOnly)
      obj.tfHideViz = tfHideOverall;
      obj.showOnlyPrimary = tfShowCurrTgtOnly;

      % Non-primary elements visibility
      onOff = onIff(~tfHideOverall && ~tfShowCurrTgtOnly);
      set(obj.hTraj,'Visible',onOff);
      set(obj.hTrx,'Visible',onOff);

      % Text labels depend on both overall visibility and showTrxIDLbl
      if obj.lObj.showTrxIDLbl
        set(obj.hTrxTxt,'Visible',onOff);
      else
        set(obj.hTrxTxt,'Visible','off');
      end

      % Primary elements visibility
      onOffPrim = onIff(~tfHideOverall);
      set(obj.hTrajPrim,'Visible',onOffPrim);
      set(obj.hTrxPrim,'Visible',onOffPrim);

      % Primary text label depends on both overall visibility and showTrxIDLbl
      if obj.lObj.showTrxIDLbl
        set(obj.hTrxTxtPrim,'Visible',onOffPrim);
      else
        set(obj.hTrxTxtPrim,'Visible','off');
      end
    end

    function setShowOnlyPrimary(obj,tf)
      obj.showOnlyPrimary = tf;
      onOff = onIff(~tf);
      set(obj.hTraj,'Visible',onOff);
      set(obj.hTrx,'Visible',onOff);
      set(obj.hTrxTxt,'Visible',onOff);
    end
    
    function set_hittest(obj,onoff)
      if ~isempty(obj.hTraj) % protect against rare cases uninitted obj (eg projLoad with "nomovie")
        [obj.hTraj.HitTest] = deal(onoff);
        [obj.hTrx.HitTest] = deal(onoff);
        [obj.hTrxTxt.HitTest] = deal(onoff);
        [obj.hTrajPrim.HitTest] = deal(onoff);
        [obj.hTrxPrim.HitTest] = deal(onoff);
        [obj.hTrxTxtPrim.HitTest] = deal(onoff);
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
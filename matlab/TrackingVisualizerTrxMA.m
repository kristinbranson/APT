classdef TrackingVisualizerTrxMA < handle
  % shows a trx/centroid marker, text label, trajectory traces
  %
  % Non-gobject model state lives on the associated
  % TrackingVisualizerTrxMAModel (accessed via obj.tvm_).

  properties
    parent_ % LabelerController reference
    tvm_ % TrackingVisualizerTrxMAModel reference, set by creator
    trxSelectCbk % function handle

    hTraj;                    % nTrx x 1 vector of line handles
    hTrx;                     % nTrx x 1 vector of line handles
    hTrxTxt;                  % nTrx x 1 vector of text handles
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

    function obj = TrackingVisualizerTrxMA(parent, tvm)
      % Construct a TrackingVisualizerTrxMA.
      %
      % parent: LabelerController
      % tvm: TrackingVisualizerTrxMAModel

      if nargin == 0
        return
      end
      obj.parent_ = parent ;
      obj.tvm_ = tvm ;
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

    function init(obj, trxSelCbk, nTrx)
      % Create gfx handles.

      obj.deleteGfxHandles();

      tvm = obj.tvm_ ;
      lObj = obj.parent_.labeler_ ; %#ok<*PROPLC>

      deleteValidGraphicsHandles(obj.hTraj);
      deleteValidGraphicsHandles(obj.hTrx);
      deleteValidGraphicsHandles(obj.hTrxTxt);
      obj.hTraj = matlab.graphics.primitive.Line.empty(0,1);
      obj.hTrx = matlab.graphics.primitive.Line.empty(0,1);
      obj.hTrxTxt = matlab.graphics.primitive.Text.empty(0,1);

      tvm.currTrx = 0 ;
      tvm.nTrxLive = 0 ;

      assert(isa(trxSelCbk,'function_handle'));
      obj.trxSelectCbk = trxSelCbk ;
      bdf = @(src,evt)obj.bdfTrx(src,evt);
      tvm.trxClickable = true ;
      clrsT = obj.setColors(nTrx);

      ax = obj.parent_.axes_curr;
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

    function clrsT = setColors(obj, nTrx)
      % init/set tvm_.clrsTrx, tvm_.clrTrxCurrent from lObj.projPrefs.Trx

      tvm = obj.tvm_ ;
      lObj = obj.parent_.labeler_ ;
      if nargin<2
        nTrx = obj.nTrx;
      end

      prefsTrx = lObj.projPrefs.Trx;
      ptcolor = prefsTrx.TrajColor;
      if ischar(ptcolor)
        clrsT = eval(ptcolor,nTrx); %#ok<EV2IN>
      else
        assert(isnumeric(ptcolor));
        nptc = size(ptcolor,1);
        nreps = ceil(nTrx/nptc);
        clrsT = repmat(ptcolor,nreps,1);
        clrsT = clrsT(1:nTrx,:);
      end
      tvm.clrsTrx = clrsT ;
      tvm.clrTrxCurrent = prefsTrx.TrajColorCurrent ;
    end

    function updateColors(obj)
      tvm = obj.tvm_ ;
      clrsT = obj.setColors();
      for iTrx = 1:obj.nTrx
        if iTrx == tvm.currTrx
          clr = tvm.clrTrxCurrent ;
        else
          clr = clrsT(iTrx,:);
        end
        obj.hTraj(iTrx).Color = clr;
        obj.hTrx(iTrx).Color = clr;
        obj.hTrxTxt(iTrx).Color = clr;
      end
    end

    function bdfTrx(obj, src, ~)
      iTrx = src.UserData;
      obj.updatePrimaryTrx(iTrx);
      obj.trxSelectCbk(iTrx);
    end

    function updatePrimaryTrx(obj, iTrxPrimary)
      % use iTrxPrimary==0 for "no current"
      tvm = obj.tvm_ ;
      tvm.currTrx = iTrxPrimary ;
      obj.updateColors();
      if tvm.showOnlyPrimary
        tfShowTrx = obj.computeTfShowTrx();
        obj.setShowTrx(tfShowTrx);
      end
    end

    function updateLiveTrx(obj, trxLive, frm, tfUpdateIDs)
      % Set/update positions of all live trx/trajs.

      tvm = obj.tvm_ ;
      lObj = obj.parent_.labeler_ ;
      nPre = tvm.showTrxPreNFrm ;
      nPst = tvm.showTrxPostNFrm ;
      pref = lObj.projPrefs.Trx;
      dx = pref.TrxIDLblOffset;

      ntrxlive = numel(trxLive);
      assert(ntrxlive<=obj.nTrx);
      tvm.nTrxLive = ntrxlive ;

      tfShowTrx = obj.computeTfShowTrx();
      obj.setShowTrx(tfShowTrx);

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

        tTraj = max(frm-nPre,t0):min(frm+nPst,t1);
        iTraj = tTraj + trxCurr.off;
        if ~isempty(trxCurr.x)
          xTraj = trxCurr.x(iTraj);
          yTraj = trxCurr.y(iTraj);
        else
          xTraj = nan;
          yTraj = nan;
        end

        set(obj.hTraj(iTrx),'XData',xTraj,'YData',yTraj);

        if lObj.showTrxIDLbl
          set(obj.hTrxTxt(iTrx),'Position',[xTrx+dx yTrx+dx 1]);
          if tfUpdateIDs
            idstr = num2str(trxCurr.id);
            set(obj.hTrxTxt(iTrx),'String',idstr);
          end
        end
      end
    end

    function setShowTrx(obj, tfShowTrx)
      lObj = obj.parent_.labeler_ ;

      set(obj.hTraj(tfShowTrx),'Visible','on');
      set(obj.hTraj(~tfShowTrx),'Visible','off');
      set(obj.hTrx(tfShowTrx),'Visible','on');
      set(obj.hTrx(~tfShowTrx),'Visible','off');
      set(obj.hTraj(tfShowTrx),'HitTest','on');
      set(obj.hTraj(~tfShowTrx),'HitTest','off');
      set(obj.hTrx(tfShowTrx),'HitTest','on');
      set(obj.hTrx(~tfShowTrx),'HitTest','off');

      if lObj.showTrxIDLbl
        set(obj.hTrxTxt(tfShowTrx),'Visible','on');
        set(obj.hTrxTxt(~tfShowTrx),'Visible','off');
        set(obj.hTrxTxt(tfShowTrx),'HitTest','on');
        set(obj.hTrxTxt(~tfShowTrx),'HitTest','off');
      else
        set(obj.hTrxTxt,'Visible','off');
      end
    end

    function tfShowTrx = computeTfShowTrx(obj)
      tvm = obj.tvm_ ;
      ntrxlive = tvm.nTrxLive ;
      tfShowTrx = false(obj.nTrx,1);
      if tvm.tfHideViz
        % none
      elseif tvm.showOnlyPrimary
        cTrx = tvm.currTrx ;
        if cTrx>0 && cTrx<=ntrxlive
          tfShowTrx(cTrx) = true;
        end
      else
        tfShowTrx(1:ntrxlive) = true;
      end
    end

    function updateShowHideAll(obj)
      tfShow = obj.computeTfShowTrx();
      obj.setShowTrx(tfShow);
    end

    function setHideViz(obj, tf)
      obj.tvm_.tfHideViz = tf ;
      obj.updateShowHideAll();
    end

    function setAllShowHide(obj, tfHideOverall, tfShowCurrTgtOnly)
      tvm = obj.tvm_ ;
      tvm.tfHideViz = tfHideOverall ;
      tvm.showOnlyPrimary = tfShowCurrTgtOnly ;
      obj.updateShowHideAll();
    end

    function setShowOnlyPrimary(obj, tf)
      obj.tvm_.showOnlyPrimary = tf ;
      obj.updateShowHideAll();
    end

    function set_hittest(obj, onoff)
      if ~isempty(obj.hTraj)
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

  end

end

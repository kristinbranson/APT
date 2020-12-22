classdef TrackingVisualizerTrx < handle
  % shows a trx/centroid marker, text label, trajectory traces
  
  properties
    lObj
    
    hTraj;                    % nTrx x 1 vector of line handles
    hTrx;                     % nTrx x 1 vector of line handles
    hTrxTxt;                  % nTrx x 1 vector of text handles
    showTrxPreNFrm = 15;      % number of preceding frames to show in traj
    showTrxPostNFrm = 5;      % number of following frames to show in traj
    
    click2nav = false;
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
    
    function obj = TrackingVisualizerTrx(labeler)
      obj.lObj = labeler;
    end
    function deleteGfxHandles(obj)
      deleteValidHandles(obj.hTraj);
      obj.hTraj = [];
      deleteValidHandles(obj.hTrx);
      obj.hTrx = [];
      deleteValidHandles(obj.hTrxTxt);
      obj.hTrxTxt = [];
    end
    
    function initShowTrx(obj,click2nav,nTrx)
      % create gfx handles
      
      obj.deleteGfxHandles();
      
      lObj = obj.lObj; %#ok<*PROPLC>
              
      deleteValidHandles(obj.hTraj);
      deleteValidHandles(obj.hTrx);
      deleteValidHandles(obj.hTrxTxt);
      obj.hTraj = matlab.graphics.primitive.Line.empty(0,1);
      obj.hTrx = matlab.graphics.primitive.Line.empty(0,1);
      obj.hTrxTxt = matlab.graphics.primitive.Text.empty(0,1);
      
      ax = lObj.gdata.axes_curr;
      pref = lObj.projPrefs.Trx;
      for i = 1:nTrx        
        obj.hTraj(i,1) = line(...
          'parent',ax,...
          'xdata',nan, ...
          'ydata',nan, ...
          'color',pref.TrajColor,...
          'linestyle',pref.TrajLineStyle, ...
          'linewidth',pref.TrajLineWidth, ...
          'HitTest','off',...
          'Tag',sprintf('Labeler_Traj_%d',i),...
          'PickableParts','none');

        obj.hTrx(i,1) = plot(ax,...
          nan,nan,pref.TrxMarker);
        if click2nav
          bdf = @(h,evt) lObj.clickTarget(h,evt,i);
        else
          bdf = [];
        end          
        set(obj.hTrx(i,1),...
          'Color',pref.TrajColor,...
          'MarkerSize',pref.TrxMarkerSize,...
          'LineWidth',pref.TrxLineWidth,...
          'Tag',sprintf('Labeler_Trx_%d',i),...
          'ButtonDownFcn',bdf,...
          'PickableParts','all',...
          'HitTest','on');
        if i == lObj.currTarget || ~click2nav,
          set(obj.hTrx(i,1),...
            'PickableParts','none',...
            'HitTest','off');
        end
        
        obj.hTrxTxt(i,1) = text(nan,nan,num2str(i),'Parent',ax,...
          'Color',pref.TrajColor,...
          'Fontsize',pref.TrxIDLblFontSize,...
          'Fontweight',pref.TrxIDLblFontWeight,...
          'PickableParts','none',...
          'Tag',sprintf('Labeler_TrxTxt_%d',i));
      end
      obj.click2nav = click2nav;
    end
    
    function updateTrx(obj,tfShow)
      % update coords/positions based on lObj.currFrame, .currTarget, .trx
      %
      % DO NOT call this for non-Trx projs (eg MA)!
      
      lObj = obj.lObj;
      obj.updateTrxCore(lObj.trx,lObj.currFrame,tfShow,lObj.currTarget,...
        false);
    end
    
    function updateTrxCore(obj,trxAll,frm,tfShow,iTgtPrimary,tfUpdateIDs)
      % 
      % trxAll: [ntrxshow] trx struct array; need not match obj.nTrx
      % frm: current frame
      % tfShow: [ntrxshow] logical
      % iTgtPrimary: [1] index into trxAll for current tgt
      % tfUpdateIDs: if true, trxAll must have IDs set and hTrxTxt are
      %   updated per these IDs.
      
      nPre = obj.showTrxPreNFrm;
      nPst = obj.showTrxPostNFrm;
      lObj = obj.lObj;
      pref = lObj.projPrefs.Trx;
      
      nTrx = numel(trxAll);
      for iTrx = 1:nTrx
        if ~tfShow(iTrx)
          % should already be hidden          
          continue;
        end
        
        trxCurr = trxAll(iTrx);
        t0 = trxCurr.firstframe;
        t1 = trxCurr.endframe;
        
        if t0<=frm && frm<=t1
          idx = frm+trxCurr.off;
          xTrx = trxCurr.x(idx);
          yTrx = trxCurr.y(idx);
        else
          xTrx = nan;
          yTrx = nan;
        end
        set(obj.hTrx(iTrx),'XData',xTrx,'YData',yTrx);
        
        %if tfShow(iTrx)
        tTraj = max(frm-nPre,t0):min(frm+nPst,t1); % could be empty array
        iTraj = tTraj + trxCurr.off;
        xTraj = trxCurr.x(iTraj);
        yTraj = trxCurr.y(iTraj);
        if iTrx==iTgtPrimary
          color = pref.TrajColorCurrent;
        else
          color = pref.TrajColor;
        end
        set(obj.hTraj(iTrx),'XData',xTraj,'YData',yTraj,'Color',color);
        set(obj.hTrx(iTrx),'Color',color);
        
        if lObj.showTrxIDLbl
          dx = pref.TrxIDLblOffset;
          set(obj.hTrxTxt(iTrx),'Position',[xTrx+dx yTrx+dx 1],...
            'Color',color);
          if tfUpdateIDs
            idstr = num2str(trxCurr.id+1);
            set(obj.hTrxTxt(iTrx),'String',idstr);
          end
        end
        %end
      end
      %fprintf('Time to update trx: %f\n',toc);
      
      if obj.click2nav
        set(obj.hTrx([1:iTgtPrimary-1,iTgtPrimary+1:end],1),...
          'PickableParts','all',...
          'HitTest','on');
        set(obj.hTrx(iTgtPrimary,1),...
          'PickableParts','none',...
          'HitTest','off');
      end
    end
    
    function updateRaw(obj,tgtIDs,p,iTgtPrimary)
      % update .hTrx, .hTraj etc based directly on coordinates
      %
      % tgtIDs: [ntgtstrked] array of target IDs (ie trxIDlbls)
      % p: [2*npts x ntgtstrked] landmarks
      % iTgtPrimary: [1] index into tgtIDs for the current primary tgt. Can
      %   be 0 indicating no-primary
      
      nTrx = numel(obj.hTrx);
      nTgts = numel(tgtIDs);
      if nTgts>nTrx
        warningNoTrace('Too many targets to display (%d). Truncating to %d.',...
          nTgts,nTrx);
        tgtIDs = tgtIDs(1:nTrx);
        p = p(:,1:nTrx);
        if iTgtPrimary>nTrx
          iTgtPrimary = 0;
        end
        nTgts = nTrx;
      end
    end
    
    function setShow(obj,tfShow,showTrxIDLbl)
      % tfShow: logical vec must match size of .hTrx etc
      set(obj.hTraj(tfShow),'Visible','on');
      set(obj.hTraj(~tfShow),'Visible','off');
      set(obj.hTrx(tfShow),'Visible','on');
      set(obj.hTrx(~tfShow),'Visible','off');
      
      if showTrxIDLbl
        set(obj.hTrxTxt(tfShow),'Visible','on');
        set(obj.hTrxTxt(~tfShow),'Visible','off');
      else
        set(obj.hTrxTxt,'Visible','off');
      end
    end
    
  end
  
end
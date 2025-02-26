classdef TrackingVisualizerTrx < handle
  % shows a trx/centroid marker, text label, trajectory traces
  
  properties
    lObj
    
    hTraj;                    % nTrx x 1 vector of line handles
    hTrx;                     % nTrx x 1 vector of line handles
    hTrxTxt;                  % nTrx x 1 vector of text handles
    showTrxPreNFrm = 15;      % number of preceding frames to show in traj
    showTrxPostNFrm = 5;      % number of following frames to show in traj
    
    trxClickable = false;
    trxSelectCbk;             % cbk with sig trxSelectCbk(iTrx); called when 
                              % trxClickable=true and on trx BDF
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
    
    function init(obj,click2nav,nTrx)
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
      
      if islogical(click2nav)
        if click2nav
          obj.trxSelectCbk = @(iTrx)(lObj.clickTarget(iTrx)) ;
          bdf = @(src,evt)obj.bdfTrx(src,evt);
        else
          obj.trxSelectCbk = [];
          bdf = [];
        end
        obj.trxClickable = click2nav;
      else
        assert(isa(click2nav,'function_handle'));
        obj.trxSelectCbk = click2nav;
        bdf = @(src,evt)obj.bdfTrx(src,evt);
        obj.trxClickable = true;
      end
      
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
        set(obj.hTrx(i,1),...
          'Color',pref.TrajColor,...
          'MarkerSize',pref.TrxMarkerSize,...
          'LineWidth',pref.TrxLineWidth,...
          'Tag',sprintf('Labeler_Trx_%d',i),...
          'ButtonDownFcn',bdf,...
          'UserData',i,...
          'PickableParts','all',...
          'HitTest','on');
        if i == lObj.currTarget || isempty(bdf) %% first clause questionable given multiple clients of this class
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
    end
    
    function bdfTrx(obj,src,~)
      iTrx = src.UserData;
      obj.trxSelectCbk(iTrx);
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
      % trxAll: [ntrxshow] trx struct array; ntrxshow must be <=obj.nTrx       
      % frm: current frame
      % tfShow: [ntrxshow] logical
      % iTgtPrimary: [1] index into trxAll for current tgt; can be 0 for
      %   "no-primary'
      % tfUpdateIDs: if true, trxAll must have IDs set and hTrxTxt are
      %   updated per these IDs.
      %
      % If ntrxshow<obj.nTrx, it is assumed that setShow() has been called
      % to hide the "extra" .trx
      
      nPre = obj.showTrxPreNFrm;
      nPst = obj.showTrxPostNFrm;
      lObj = obj.lObj;
      pref = lObj.projPrefs.Trx;
      
      nTrx = numel(trxAll);
      
      assert(nTrx<=obj.nTrx); 
      
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
            idstr = num2str(trxCurr.id);
            set(obj.hTrxTxt(iTrx),'String',idstr);
          end
        end
        %end
      end
      %fprintf('Time to update trx: %f\n',toc);
      
      if obj.trxClickable
        set(obj.hTrx([1:iTgtPrimary-1,iTgtPrimary+1:end],1),...
          'PickableParts','all',...
          'HitTest','on');
        if iTgtPrimary>0
          set(obj.hTrx(iTgtPrimary,1),...
            'PickableParts','none',...
            'HitTest','off');
        end
      end
    end
    
    function updatePrimary(~, ~)
      % none    
    end
    
    function setShow(obj,tfShow)
      % tfShow: logical vec must match size of .hTrx etc
      set(obj.hTraj(tfShow),'Visible','on');
      set(obj.hTraj(~tfShow),'Visible','off');
      set(obj.hTrx(tfShow),'Visible','on');
      set(obj.hTrx(~tfShow),'Visible','off');
      
     if obj.lObj.showTrxIDLbl
        set(obj.hTrxTxt(tfShow),'Visible','on');
        set(obj.hTrxTxt(~tfShow),'Visible','off');
      else
        set(obj.hTrxTxt,'Visible','off');
      end
    end
    
  end
  
end
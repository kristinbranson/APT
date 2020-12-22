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
      
      lObj = obj.lObj;
      t = lObj.currFrame;
      iTgt = lObj.currTarget;
      trxAll = lObj.trx;
      nPre = obj.showTrxPreNFrm;
      nPst = obj.showTrxPostNFrm;
      pref = lObj.projPrefs.Trx;      
      
      %tic;
      nTrx = numel(trxAll);
      for iTrx = 1:nTrx
        trxCurr = trxAll(iTrx);
        t0 = trxCurr.firstframe;
        t1 = trxCurr.endframe;
        
        if t0<=t && t<=t1
          idx = t+trxCurr.off;
          xTrx = trxCurr.x(idx);
          yTrx = trxCurr.y(idx);
        else
          xTrx = nan;
          yTrx = nan;
        end
        set(obj.hTrx(iTrx),'XData',xTrx,'YData',yTrx);
        
        if tfShow(iTrx)
          tTraj = max(t-nPre,t0):min(t+nPst,t1); % could be empty array
          iTraj = tTraj + trxCurr.off;
          xTraj = trxCurr.x(iTraj);
          yTraj = trxCurr.y(iTraj);
          if iTrx==iTgt
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
          end
        end
          
%           if tfShowEll && t0<=t && t<=t1
%             ellipsedraw(2*trxCurr.a(idx),2*trxCurr.b(idx),...
%               trxCurr.x(idx),trxCurr.y(idx),trxCurr.theta(idx),'-',...
%               'hEllipse',obj.hTrxEll(iTrx),'noseLine',true);
%           end
        %end
      end
      %fprintf('Time to update trx: %f\n',toc);
      
      if obj.click2nav
        set(obj.hTrx([1:iTgt-1,iTgt+1:end],1),...
          'PickableParts','all',...
          'HitTest','on');
        set(obj.hTrx(iTgt,1),...
          'PickableParts','none',...
          'HitTest','off');
      end
      
%       if tfShowEll
%         set(obj.hTrxEll(tfShow),'Visible','on');
%         set(obj.hTrxEll(~tfShow),'Visible','off');
%       else
%         set(obj.hTrxEll,'Visible','off');
%       end
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
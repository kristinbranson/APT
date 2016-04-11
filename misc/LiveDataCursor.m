classdef LiveDataCursor < handle
  % Shows a live cursor on a set of subplotaxes in a single/common fig
  % Hooks for WBMF, AXBDF
  
  properties
    hFig
    hAx % vector of axes
    tfAxUse % [Nax] logical, 1 if axis has data
    
    iAxSel % currently selected axis. 
    hLine % original plot data
    tStrs % original title strs
    hPt % highlight pt. only shown in currently selected axis
    hCrossHair; % highlight line. shown on all axes
    
    yvals % [Nax] cell array of vectors of original y data for each axis
    bdfCbks % [Nax] cell array of bdf callbacks. These are called when an 
            % axis that is currently selected is clicked. Sig:
            % fcn(xsel,ysel) where (xsel,ysel) is the data pt nearest to
            % where user clicked.
  end
  
  methods
    function obj = LiveDataCursor(hF,hA,bdfCbks)
      % hF: figure handle
      % hA: [Nax] vector of axes handles
      % bdfCbks: [Nax] cell array of bdf callbacks 
      
      tStrs = cell(size(hA));
      hL = gobjects(size(hA));
      hP = gobjects(size(hA));
      hCH = gobjects(size(hA));
      yvals = cell(size(hA));
      tfAxUse = false(size(hA));
      for i = 1:numel(hA)
        ax = hA(i);
        hold(ax,'on');
        
        ax.Box = 'on';
        ax.LineWidth = 1;
        ax.UserData = i;
        ax.ButtonDownFcn = @(s,e)obj.axbdf(s,e);
        
        tmp = findobj(ax,'type','line');
        tfAxUse(i) = ~isempty(tmp);
        if tfAxUse(i)
          tStrs{i} = ax.Title.String;
          hL(i) = tmp;
          hP(i) = plot(ax,nan,nan,'ro','markerfacecolor',[1 0 0]);
          x0 = min(hL(i).XData);
          x1 = max(hL(i).XData);
          hCH(i) = plot(ax,[x0 x1],[nan nan],'r-');
          
          yvals{i} = hL(i).YData;
        end
      end
      
      hF.WindowButtonMotionFcn = @(s,e)obj.wbmf();      
      obj.hFig = hF;
      obj.hAx = hA;
      obj.tfAxUse = tfAxUse;
      obj.hLine = hL;
      obj.tStrs = tStrs;
      obj.hPt = hP;
      obj.hCrossHair = hCH;
      
      obj.yvals = yvals;
      obj.bdfCbks = bdfCbks;
      
      hF.UserData = obj;
      
      if nnz(tfAxUse)==1
        obj.axsel(find(tfAxUse)); %#ok<FNDSB>
      end
    end
    function axsel(obj,i)
      iold = obj.iAxSel;
      if ~isempty(iold)
        obj.hAx(iold).LineWidth = 1;
        obj.hAx(iold).Title.String = obj.tStrs{iold};
        obj.hPt(iold).XData = nan;
%        obj.hCrossHair(iold).YData = [nan nan];
      end
      obj.hAx(i).LineWidth = 2;
      obj.iAxSel = i;
    end
    function axbdf(obj,src,~)
      i = src.UserData;
      if i==obj.iAxSel
        pt = src.CurrentPoint(1,:);
        x = pt(1);
        
        hL = obj.hLine(i);
        d = abs(x-hL.XData);
        idx = argmin(d);
        x = hL.XData(idx);
        y = hL.YData(idx);
        obj.bdfCbks{i}(x,y);
      else
        obj.axsel(i);
      end
    end
    
    function wbmf(obj)
      ax = obj.hAx;
      i = obj.iAxSel;
      if ~isempty(i)
        hL = obj.hLine(i);
        hP = obj.hPt(i);
        
        pt = ax(i).CurrentPoint(1,:);
        x = pt(1);
        d = abs(x-hL.XData);
        idx = argmin(d);
        x = hL.XData(idx);
        y = hL.YData(idx);
        
        hP.XData = x;
        hP.YData = y;
        
        iAxUse = find(obj.tfAxUse);        
        for iAx = iAxUse(:)'          
          hCH = obj.hCrossHair(iAx);
          hCH.YData = [y y];
          
          yy = obj.yvals{iAx};
          nTot = numel(yy);
          nUse = nnz(yy>y);
          str = sprintf('%s: y=%.3g. nUse/nTot=%d/%d (%d%%)',obj.tStrs{i},...
            y,nUse,nTot,round(nUse/nTot*100));  
          ax(iAx).Title.String = str;
        end
      end
    end
  end
end

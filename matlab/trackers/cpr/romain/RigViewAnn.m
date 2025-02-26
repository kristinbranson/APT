classdef RigViewAnn < handle
  % Plot Annotations for multi-view labeling
  %
  % This is mostly just a passive struct (+ a little lifetime management)
  
  properties    
    ax
    color
    
    anchorIAx; % axis/cam index of anchor pt
    anchorHIMP; % impoint for anchor pt
    epiIAx; % [2] axis indices for other two views/axes
    epiHLines; % [2] epipolar lines handles in other two axes
    secondIAx; % axis/cam index of 2nd pt
    secondHIMP; % impoint for 2nd pt
    thirdIAx; % axis/cam index for 3rd pt
    thirdHPt; % hLine handle for 3rd pt (3 pts)
    
    markersz = 10;
  end
  
  methods
    
    function obj = RigViewAnn(axs,clr,ancIAx,ancH)
      obj.ax = axs;
      obj.color = clr;     
          
      obj.anchorIAx = ancIAx;
      obj.anchorHIMP = ancH;
      
      hCirc = findall(ancH,'Tag','circle');
      hCirc.MarkerSize = obj.markersz;
      hCirc.MarkerFaceColor = clr;
      hPlus = findall(ancH,'Tag','plus');
      hPlus.Marker = 'none';
      
      iAx1 = mod(ancIAx,3)+1;
      iAx2 = mod(ancIAx+1,3)+1;
      hold(axs(iAx1),'on');
      hold(axs(iAx2),'on');
      obj.epiIAx = [iAx1 iAx2];
      obj.epiHLines = [...
        plot(axs(iAx1),nan,nan,'-','Color',clr,'LineWidth',2) ...
        plot(axs(iAx2),nan,nan,'-','Color',clr,'LineWidth',2)];
      obj.epiHLines(1).UserData = obj;
      obj.epiHLines(2).UserData = obj;
      
      obj.secondIAx = 0;
      obj.secondHIMP = [];
      obj.thirdIAx = 0;
      obj.thirdHPt = [];
    end
    
    function delete(obj)
      delete(obj.anchorHIMP);
      obj.anchorHIMP = [];
      deleteValidGraphicsHandles(obj.epiHLines);
      obj.epiHLines = [];
      if ~isempty(obj.secondHIMP);
        delete(obj.secondHIMP);
        obj.secondHIMP = [];
      end
      deleteValidGraphicsHandles(obj.thirdHPt);
      obj.thirdHPt = [];
    end  
    
    function addSecondPt(obj,iAx2,hIMP2)
      assert(isempty(obj.secondHIMP));
      assert(isempty(obj.thirdHPt));
      
      h = findall(hIMP2,'Tag','circle');
      h.Marker = 's';
      h.MarkerSize = obj.markersz;
      h.MarkerFaceColor = obj.color;
      hPlus = findall(hIMP2,'Tag','plus');
      hPlus.Marker = 'none';
      
      obj.secondHIMP = hIMP2;
      hIMP2.deleteFcn = @()obj.rmSecondPt();
      obj.secondIAx = iAx2;
      
      obj.thirdIAx = setdiff(1:3,[iAx2 obj.anchorIAx]);
      obj.thirdHPt = plot(obj.ax(obj.thirdIAx),[nan nan nan],[nan nan nan],...
        'x','Color',obj.color,'LineWidth',4);
    end
       
    function rmSecondPt(obj)
      delete(obj.secondHIMP);
      obj.secondHIMP = [];
      obj.secondIAx = 0;
      deleteValidGraphicsHandles(obj.thirdHPt);
      obj.thirdHPt = [];
      obj.thirdIAx = 0;      
    end
    
  end
  
end
classdef ParameterVisualizationTgtCropRadius < ParameterVisualization
  
  properties
    % If true, a prop for this pvObj is currently selected, and we are 
    % successfully initted/displaying something.
    initSuccessful = false; 
    
    hRect % scalar handle to rectangle. set/created during init
    xTrx % xTrx/yTrx: (x,y) for trx center. set/created during init 
    yTrx
    hRectArgs = {'EdgeColor','r','LineWidth',2}; 
  end
  
  methods
    
    function propSelected(obj,hAx,lObj,propFullName,sPrm)
      
      obj.initSuccessful = false;
      if ~lObj.hasMovie
        ParameterVisualization.grayOutAxes(hAx,'No movie available.');
        return;
      end      
      if ~lObj.hasTrx
        ParameterVisualization.grayOutAxes(hAx,'Project does not have trx.');
        return;
      end
      
      gdata = lObj.gdata;
      im = gdata.image_curr;
      im = im.CData;
      
      rad0 = sPrm.ROOT.ImageProcessing.MultiTarget.TargetCrop.Radius;

      iMov = lObj.currMovie;
      frm = lObj.currFrame;
      iTgt = lObj.currTarget;
      trx = lObj.currTrx;
      [obj.xTrx,obj.yTrx] = readtrx(trx,frm,1);
      rectPos = obj.getRectPos(rad0);
      
      cla(hAx);
      hold(hAx,'off');
      imshow(im,'Parent',hAx);
      hold(hAx,'on');
      axis(hAx,'image');
      colormap(hAx,'gray');
      caxis(hAx,'auto');
%       axis(hAx,'auto');
      tstr = 'Movie images will be cropped as shown for tracking';
      title(hAx,tstr,'interpreter','none','fontweight','normal',...
        'fontsize',10);
      deleteValidHandles(obj.hRect);
      obj.hRect = rectangle('Position',rectPos,obj.hRectArgs{:});
      
      obj.initSuccessful = true;
    end
    
    function propUnselected(obj)
      deleteValidHandles(obj.hRect);
      obj.hRect = [];
      obj.initSuccessful = false;
    end

    function propUpdated(obj,hAx,lObj,propFullName,sPrm)
      if obj.initSuccessful
        rad = sPrm.ROOT.ImageProcessing.MultiTarget.TargetCrop.Radius;
        rectPos = obj.getRectPos(rad);
        obj.hRect.Position = rectPos;
      end
    end
    
    function propUpdatedDynamic(obj,hAx,lObj,propFullName,sPrm,rad)
      if obj.initSuccessful
        rectPos = obj.getRectPos(rad);
        obj.hRect.Position = rectPos;
      end
    end
    
    function rectPos = getRectPos(obj,rad)
      rectW = 2*rad+1;
      rectPos = [obj.xTrx-rad obj.yTrx-rad rectW rectW];
    end
    
  end
  
end
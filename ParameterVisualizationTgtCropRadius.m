classdef ParameterVisualizationTgtCropRadius < ParameterVisualization
  
  properties
    initSuccessful = false;
    
    hRect % scalar handle to rectangle. set/created during init
    xTrx % xTrx/yTrx: (x,y) for trx center. set/created during init 
    yTrx
    hRectArgs = {'EdgeColor','r','LineWidth',2}; 
  end
  
  methods
    
    function init(obj,hAx,lObj,propFullName,sPrm)
      
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
      
      rad0 = sPrm.ROOT.Track.MultiTarget.TargetCrop.Radius;

      iMov = lObj.currMovie;
      frm = lObj.currFrame;
      iTgt = lObj.currTarget;
      trx = lObj.currTrx;
      [obj.xTrx,obj.yTrx] = readtrx(trx,frm,1);
      rectPos = obj.getRectPos(rad0);
      
      cla(hAx);
      imshow(im,'Parent',hAx);
      caxis(hAx,'auto');
      tstr = sprintf('movie %d, frame %d, target %d',iMov,frm,iTgt);
      title(hAx,tstr,'interpreter','none','fontweight','normal');
      deleteValidHandles(obj.hRect);
      obj.hRect = rectangle('Position',rectPos,obj.hRectArgs{:});
      
      obj.initSuccessful = true;
    end

    function update(obj,hAx,lObj,propFullName,sPrm)      
      if obj.initSuccessful
        rad = sPrm.ROOT.Track.MultiTarget.TargetCrop.Radius;
        rectPos = obj.getRectPos(rad);
        obj.hRect.Position = rectPos;
      end
    end
    
    function updateNewVal(obj,hAx,lObj,propFullName,sPrm,rad)
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
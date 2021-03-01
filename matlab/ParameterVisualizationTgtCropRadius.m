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
    
    function isOk = plotOk(obj)
      isOk = ~isempty(obj.hRect) && ishandle(obj.hRect);
    end
    
    function propSelected(obj,hAx,lObj,propFullName,sPrm)      
      obj.init(hAx,lObj,propFullName,sPrm);
    end
    
    function init(obj,hAx,lObj,propFullName,sPrm)
      
      obj.initSuccessful = false;
      set(hAx,'Units','normalized','Position',obj.axPos);
      
      if ~lObj.hasMovie
        ParameterVisualization.grayOutAxes(hAx,'No movie available.');
        return;
      end
      
      % Set .xTrx, .yTrx; get im
      if lObj.hasTrx
        frm = lObj.currFrame;
        trx = lObj.currTrx;
        [obj.xTrx,obj.yTrx] = readtrx(trx,frm,1);
        gdata = lObj.gdata;
        im = gdata.image_curr;
        im = im.CData;
        tstr = 'Movie images will be cropped as shown for tracking';
      elseif lObj.maIsMA
        [tffound,mIdx,frm,~,xyLbl] = lObj.labelFindOneLabeledFrame();
        if ~tffound
          ParameterVisualization.grayOutAxes(hAx,...
            'Visualization unavailable until at least one animal is labeled.');
          return;
        end        
        mr = MovieReader;
        assert(~lObj.isMultiView);
        IVIEW = 1;
        lObj.movieMovieReaderOpen(mr,mIdx,IVIEW);
        im = mr.readframe(frm);
        xycent = nanmean(xyLbl,1);
        obj.xTrx = xycent(1);
        obj.yTrx = xycent(2);
        tstr = 'Region within ROI used during training';
      else
        ParameterVisualization.grayOutAxes(hAx,'Project is single-animal.');
        return;
      end
      
      rad0 = sPrm.ROOT.ImageProcessing.MultiTarget.TargetCrop.Radius;      
      rectPos = obj.getRectPos(rad0);
          
      cla(hAx);
      hold(hAx,'off');
      imshow(im,'Parent',hAx);
      hold(hAx,'on');
      axis(hAx,'image');
      colormap(hAx,'gray');
      caxis(hAx,'auto');
%       axis(hAx,'auto');
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
      if obj.initSuccessful && obj.plotOk(),
        rad = sPrm.ROOT.ImageProcessing.MultiTarget.TargetCrop.Radius;
        rectPos = obj.getRectPos(rad);
        obj.hRect.Position = rectPos;
      else
        obj.init(hAx,lObj,propFullName,sPrm);
      end
    end
    
    function propUpdatedDynamic(obj,hAx,lObj,propFullName,sPrm,rad)
      if obj.initSuccessful && obj.plotOk(),
        rectPos = obj.getRectPos(rad);
        obj.hRect.Position = rectPos;
      else
        obj.init(hAx,lObj,propFullName,sPrm);
      end
    end
    
    function rectPos = getRectPos(obj,rad)
      rectW = 2*rad+1;
      rectPos = [obj.xTrx-rad obj.yTrx-rad rectW rectW];
    end
    
  end
  
end
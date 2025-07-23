classdef ParameterVisualizationTgtCropRadius < ParameterVisualization
  
  properties
    % If true, a prop for this pvObj is currently selected, and we are 
    % successfully initted/displaying something.
    initSuccessful = false; 
    
    hRect % scalar line handle. set/created during init
    
    isMA % scalar logical
    
    % used for non-MA
    xTrx % xTrx/yTrx: (x,y) for trx center. set/created during init 
    yTrx
    
    % used for MA
    xyLbl % [npts x 2]
    
    hRectArgs = {'Color','r','LineWidth',2}; 
  end
  
  methods
    
    function isOk = plotOk(obj)
      isOk = ~isempty(obj.hRect) && ishandle(obj.hRect);
    end
    
    function propSelected(obj,hAx,lObj,propFullName,prm)      
      obj.init(hAx,lObj,propFullName,prm);
    end
    
    function init(obj,hAx,lObj,propFullName,prm)
      
      obj.initSuccessful = false;
      if ~strcmp(hAx.Parent.Type,'tiledlayout'),
        set(hAx,'Units','normalized','Position',obj.axPos);
      end
      
      if ~lObj.hasMovie
        ParameterVisualization.grayOutAxes(hAx,'No movie available.');
        return;
      end
      
      % Set .xTrx, .yTrx; get im
      obj.isMA = lObj.maIsMA;
      if lObj.hasTrx
        frm = lObj.currFrame;
        trx = lObj.currTrx;
        [obj.xTrx,obj.yTrx] = readtrx(trx,frm,1);
        obj.xyLbl = [];
        gdata = lObj.gdata;
        im = gdata.image_curr;
        im = im.CData;
        tstr = 'Movie images will be cropped as shown for tracking';
      elseif lObj.maIsMA
        [tffound,mIdx,frm,~,xyLbl] = lObj.labelFindOneLabeledFrame(); %#ok<PROPLC>
        if ~tffound
          ParameterVisualization.grayOutAxes(hAx,...
            'Visualization unavailable until at least one animal is labeled.');
          return;
        end        
        mr = MovieReader();
        assert(~lObj.isMultiView);
        IVIEW = 1;
        mr.openForLabeler(lObj,mIdx,IVIEW);
        im = mr.readframe(frm);
        
        obj.xyLbl = xyLbl; %#ok<PROPLC>
        obj.xTrx = [];
        obj.yTrx = [];
        tstr = 'Region within ROI used during training';
      else
        ParameterVisualization.grayOutAxes(hAx,'Project is single-animal.');
        return;
      end
      
      sPrm_MultiTgt_TargetCrop = ParameterVisualizationTgtCropRadius.getParamValue(prm,'ROOT.MultiAnimal.TargetCrop');     
      rectPos = obj.getRectPos(lObj,sPrm_MultiTgt_TargetCrop);
          
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
      deleteValidGraphicsHandles(obj.hRect);
      obj.hRect = plot(hAx,rectPos(:,1),rectPos(:,2),obj.hRectArgs{:});
      
      obj.initSuccessful = true;
    end
    
    function propUnselected(obj)
      deleteValidGraphicsHandles(obj.hRect);
      obj.hRect = [];
      obj.initSuccessful = false;
    end

    function propUpdated(obj,hAx,lObj,propFullName,prm)
      if obj.initSuccessful && obj.plotOk(),
        sPrm_MultiTgt_TargetCrop = ParameterVisualizationTgtCropRadius.getParamValue(prm,'ROOT.MultiAnimal.TargetCrop'); 
        rectPos = obj.getRectPos(lObj,sPrm_MultiTgt_TargetCrop);
        set(obj.hRect,'XData',rectPos(:,1),'YData',rectPos(:,2));
      else
        obj.init(hAx,lObj,propFullName,prm);
      end
    end
    
    function propUpdatedDynamic(obj,hAx,lObj,propFullName,prm,val)
      propFullName = ParameterVisualizationTgtCropRadius.modernizePropName(propFullName);
      if obj.initSuccessful && obj.plotOk()
        sPrm_MultiTgt_TargetCrop = ParameterVisualizationTgtCropRadius.getParamValue(prm,'ROOT.MultiAnimal.TargetCrop');
        assert(startsWith(propFullName,'ROOT.MultiAnimal.TargetCrop.'));
        toks = strsplit(propFullName,'.');        
        propShort = toks{end};
        sPrm_MultiTgt_TargetCrop.(propShort) = val;        
        rectPos = obj.getRectPos(lObj,sPrm_MultiTgt_TargetCrop);
        set(obj.hRect,'XData',rectPos(:,1),'YData',rectPos(:,2));
      else
        obj.init(hAx,lObj,propFullName,prm);
      end
    end
    
    function rectPos = getRectPos(obj,lObj,sPrm)
      % rectPos: [c x 2] col1 is [x1;x2;x3;x4;x5]; col2 is [y1;y2; etc].
      
      rad = maGetTgtCropRad(sPrm);
      if obj.isMA
        xyc = nanmean(obj.xyLbl,1);
        xc = xyc(1);
        yc = xyc(2);
        %rectPos = lObj.maGetRoi(obj.xyLbl,sPrm);
      else
        xc = obj.xTrx;
        yc = obj.yTrx;
      end
      
      x0 = xc-rad;
      x1 = xc+rad;
      y0 = yc-rad;
      y1 = yc+rad;
      rectPos = [x0 x0 x1 x1;y0 y1 y1 y0].';
      
      % for plotting
      rectPos(5,:) = rectPos(1,:);
    end
    
  end
  
end
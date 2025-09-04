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
    
    function init(obj,hTile,lObj,propFullName,prm,varargin)
      
      if nargin > 1,
        init@ParameterVisualization(obj,hTile,lObj,propFullName,prm);
      end
      if isempty(obj.hAx),
        obj.hAx = nexttile(obj.hTile);
      end

      if ~obj.lObj.hasMovie
        ParameterVisualizationTgtCropRadius.grayOutAxes('No movie available.');
        return;
      end
     
      % Set .xTrx, .yTrx; get im
      obj.isMA = obj.lObj.maIsMA;
      if obj.lObj.hasTrx
        frm = obj.lObj.currFrame;
        trx = obj.lObj.currTrx;
        [obj.xTrx,obj.yTrx] = readtrx(trx,frm,1);
        obj.xyLbl = [];
        gdata = obj.lObj.gdata;
        im = gdata.image_curr;
        im = im.CData;
        tstr = 'Movie images will be cropped as shown for tracking';
      elseif obj.lObj.maIsMA
        [tffound,mIdx,frm,~,xyLbl] = obj.lObj.labelFindOneLabeledFrame(); %#ok<PROPLC>
        if ~tffound
          ParameterVisualization.grayOutAxes('Visualization unavailable until at least one animal is labeled.');
          return;
        end        
        mr = MovieReader();
        assert(~obj.lObj.isMultiView);
        IVIEW = 1;
        mr.openForLabeler(lObj,mIdx,IVIEW);
        im = mr.readframe(frm);
        
        obj.xyLbl = xyLbl; %#ok<PROPLC>
        obj.xTrx = [];
        obj.yTrx = [];
        tstr = 'Region within ROI used during training';
      else
        ParameterVisualization.grayOutAxes('Project is single-animal.');
        return;
      end
      
      rectPos = obj.getRectPos();
          
      cla(obj.hAx);
      hold(obj.hAx,'off');
      imshow(im,'Parent',obj.hAx);
      hold(obj.hAx,'on');
      axis(obj.hAx,'image');
      colormap(obj.hAx,'gray');
      clim(obj.hAx,'auto');
%       axis(obj.hAx,'auto');
      title(obj.hAx,tstr,'interpreter','none','fontweight','normal',...
        'fontsize',10);
      deleteValidGraphicsHandles(obj.hRect);
      obj.hRect = plot(obj.hAx,rectPos(:,1),rectPos(:,2),obj.hRectArgs{:});
      
      obj.initSuccessful = true;
    end
    
    function clear(obj)
      clear@ParameterVisualization(obj);
      obj.initSuccessful = false;
    end

    function update(obj)
      if obj.initSuccessful && obj.plotOk(),
        rectPos = obj.getRectPos();
        set(obj.hRect,'XData',rectPos(:,1),'YData',rectPos(:,2));
      else
        obj.init();
      end
    end
    
    function rectPos = getRectPos(obj)
      % rectPos: [c x 2] col1 is [x1;x2;x3;x4;x5]; col2 is [y1;y2; etc].
      
      rad = APTParameters.maGetTgtCropRad(obj.prm);
      if obj.isMA
        xyc = mean(obj.xyLbl,1,'omitmissing');
        xc = xyc(1);
        yc = xyc(2);
        %rectPos = obj.lObj.maGetRoi(obj.xyLbl,sPrm);
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
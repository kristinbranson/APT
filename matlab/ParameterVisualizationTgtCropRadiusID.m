classdef ParameterVisualizationTgtCropRadiusID < ParameterVisualization
  % Visualizes the crop used for identity linking: a labeled animal,
  % aligned head-up when head/tail landmarks are specified, with the
  % link_id_cropsz_height/width rectangle overlaid.

  properties
    % If true, a prop for this pvObj is currently selected, and we are
    % successfully initted/displaying something.
    initSuccessful = false;

    hRect % scalar line handle. set/created during init

    xyLbl % [npts x 2] labels of the animal being displayed

    % head-tail alignment
    hasHT       % scalar logical, true if skelHead/skelTail are defined
    imFull      % full movie frame the displayed crop is cut from
    bodyCtr     % [1 x 2] body center (x,y) in full-frame coordinates
    htAngle     % head->tail angle used for the aligned crop, radians

    hRectArgs = {'Color','r','LineWidth',2};
  end

  methods

    function isOk = plotOk(obj)
      isOk = ~isempty(obj.hRect) && ishandle(obj.hRect);
    end

    function init(obj,hTile,lObj,propFullName,prm,varargin)
      % Read one labeled frame, align it head-up if possible, and draw
      % the ID-linking crop rectangle.
      if nargin > 1,
        init@ParameterVisualization(obj,hTile,lObj,propFullName,prm);
      end
      if isempty(obj.hAx),
        obj.hAx = nexttile(obj.hTile);
      end
      obj.initSuccessful = false;

      if ~obj.lObj.hasMovie
        obj.grayOutAxes('No movie available.');
        return;
      end
      if ~obj.lObj.maIsMA
        obj.grayOutAxes('Project is single-animal.');
        return;
      end

      [tffound,mIdx,frm,~,xyLbl] = obj.lObj.labelFindOneLabeledFrame(); %#ok<PROPLC>
      if ~tffound
        obj.grayOutAxes('Visualization unavailable until at least one animal is labeled.');
        return;
      end
      mr = MovieReader();
      assert(~obj.lObj.isMultiView);
      IVIEW = 1;
      mr.openForLabeler(obj.lObj,mIdx,IVIEW);
      obj.imFull = mr.readframe(frm);
      obj.xyLbl = xyLbl; %#ok<PROPLC>

      % Align image using head-tail landmarks if available.
      % The head is mapped to the +y direction (downward in image coords).
      obj.hasHT = ~isempty(obj.lObj.skelHead) && ~isempty(obj.lObj.skelTail);
      if obj.hasHT
        hd = obj.xyLbl(obj.lObj.skelHead,:);   % [1 x 2] (x,y)
        tl = obj.xyLbl(obj.lObj.skelTail,:);   % [1 x 2] (x,y)
        obj.bodyCtr = (hd + tl) / 2;           % [1 x 2] (x,y)
        obj.htAngle = atan2(tl(2)-hd(2), hd(1)-tl(1));
      else
        obj.bodyCtr = [];
        obj.htAngle = 0;
      end

      obj.redraw_();
      obj.initSuccessful = true;
    end

    function clear(obj)
      clear@ParameterVisualization(obj);
      obj.hRect = [];
      obj.initSuccessful = false;
    end

    function update(obj)
      if obj.initSuccessful && obj.plotOk(),
        obj.redraw_();
      else
        obj.init();
      end
    end

    function redraw_(obj)
      % Redraw the (possibly aligned) crop image and the crop rectangle
      % from the current parameter values.
      [hh,ww] = obj.getCropSize_();

      if obj.hasHT
        cropSize = max(hh,ww)*2;
        im = CropImAroundTrx(obj.imFull,obj.bodyCtr(1),obj.bodyCtr(2),...
                             -obj.htAngle,cropSize,cropSize);
        imCtr = size(im,1,2)/2;
        xc = imCtr(1);
        yc = imCtr(2);
        halfWidth = ww/2;
        halfHeight = hh/2;
        tstr = 'Aligned crop region used during ID tracking';
      else
        % No head-tail: fall back to the label centroid and a square crop
        im = obj.imFull;
        xyc = mean(obj.xyLbl,1,'omitmissing');
        xc = xyc(1);
        yc = xyc(2);
        rad = APTParameters.getMATargetCropRadiusManual(obj.prm);
        halfWidth = rad;
        halfHeight = rad;
        tstr = 'Crop region used during ID tracking';
      end

      x0 = xc-halfWidth;
      x1 = xc+halfWidth;
      y0 = yc-halfHeight;
      y1 = yc+halfHeight;
      rectPos = [x0 x0 x1 x1;y0 y1 y1 y0].';
      rectPos(5,:) = rectPos(1,:);  % close the rectangle for plotting

      cla(obj.hAx);
      hold(obj.hAx,'off');
      imshow(im,'Parent',obj.hAx);
      hold(obj.hAx,'on');
      axis(obj.hAx,'image');
      colormap(obj.hAx,'gray');
      clim(obj.hAx,'auto');
      title(obj.hAx,tstr,'interpreter','none','fontweight','normal',...
        'fontsize',10);
      deleteValidGraphicsHandles(obj.hRect);
      obj.hRect = plot(obj.hAx,rectPos(:,1),rectPos(:,2),obj.hRectArgs{:});
    end

    function [hh,ww] = getCropSize_(obj)
      % Read the ID-linking crop height/width from the parameter tree.
      hh = APTParameters.getParam(obj.prm,...
        'ROOT.MultiAnimal.Track.TrackletStitch.link_id_cropsz_height');
      ww = APTParameters.getParam(obj.prm,...
        'ROOT.MultiAnimal.Track.TrackletStitch.link_id_cropsz_width');
      if hh <= 0 || ww <= 0
        % Negative means "use the automatically computed crop size"; fall
        % back to the manual target-crop radius for display.
        rad = APTParameters.getMATargetCropRadiusManual(obj.prm);
        if hh <= 0, hh = 2*rad; end
        if ww <= 0, ww = 2*rad; end
      end
    end

  end

end

classdef ParameterVisualizationTgtCropRadiusID < ParameterVisualization

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

    % head-tail alignment
    hasHT       % scalar logical, true if skelHead/skelTail are defined
    rotAngleDeg % rotation applied to the displayed image (degrees CCW, per imrotate)
    bodyCtrRot  % [1 x 2] body center (x,y) in rotated image coordinates

    hRectArgs = {'Color','r','LineWidth',2};
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
      if lObj.maIsMA
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
        tstr = 'Aligned crop region used during ID tracking';
      else
        ParameterVisualization.grayOutAxes(hAx,'Project is single-animal.');
        return;
      end

      % Align image using head-tail landmarks if available.
      % The head is mapped to the +y direction (downward in image coords).
      obj.hasHT = ~isempty(lObj.skelHead) && ~isempty(lObj.skelTail);
      obj.rotAngleDeg = 0;
      obj.bodyCtrRot = [];

      hh = sPrm.ROOT.MultiAnimal.Track.TrackletStitch.link_id_cropsz_height;
      ww = sPrm.ROOT.MultiAnimal.Track.TrackletStitch.link_id_cropsz_width;

      if obj.hasHT && ~isempty(obj.xyLbl)
        hd = obj.xyLbl(lObj.skelHead,:);   % [1 x 2] (x,y)
        tl = obj.xyLbl(lObj.skelTail,:);   % [1 x 2] (x,y)
        body_ctr = (hd + tl) / 2;          % [1 x 2] (x,y)

        % ht_angle: angle of the head->tail vector in image coords.
        % Rotating by -(pi/2 + ht_angle) maps the head to +y (downward).
        ht_angle = atan2(tl(2)-hd(2), hd(1)-tl(1));
        obj.rotAngleDeg = (pi/2 - ht_angle) * 180/pi;

        % Record original image center before rotation
        [nr_orig, nc_orig] = size(im, 1, 2);
        im_ctr_orig = [(nc_orig+1)/2, (nr_orig+1)/2];  % (x,y)

        % Rotate image; 'loose' preserves the full rotated extent
        % im = imrotate(im, obj.rotAngleDeg, 'bilinear', 'loose');
        crop_sz = max(hh,ww)*2;
        im = CropImAroundTrx(im,body_ctr(1),body_ctr(2),-ht_angle,crop_sz,crop_sz);

        obj.bodyCtrRot = size(im,1,2)/2;
      end

      sPrm_MultiTgt_TargetCrop = [ww,hh];
      % sPrm_MultiTgt_TargetCrop = sPrm.ROOT.MultiAnimal.Track.TrackletStitch.link_id_cropsz;
      rectPos = obj.getRectPos(lObj,sPrm_MultiTgt_TargetCrop);

      cla(hAx);
      hold(hAx,'off');
      imshow(im,'Parent',hAx);
      hold(hAx,'on');
      axis(hAx,'image');
      colormap(hAx,'gray');
      caxis(hAx,'auto');
      title(hAx,tstr,'interpreter','none','fontweight','normal',...
        'fontsize',10);
      deleteValidGraphicsHandles(obj.hRect);
      obj.hRect = plot(rectPos(:,1),rectPos(:,2),obj.hRectArgs{:});

      obj.initSuccessful = true;
    end

    function propUnselected(obj)
      deleteValidGraphicsHandles(obj.hRect);
      obj.hRect = [];
      obj.initSuccessful = false;
    end

    function propUpdated(obj,hAx,lObj,propFullName,sPrm)
      if obj.initSuccessful && obj.plotOk(),
        sPrm_MultiTgt_TargetCrop = sPrm.ROOT.MultiAnimal.TargetCrop;
        rectPos = obj.getRectPos(lObj,sPrm_MultiTgt_TargetCrop);
        set(obj.hRect,'XData',rectPos(:,1),'YData',rectPos(:,2));
      else
        obj.init(hAx,lObj,propFullName,sPrm);
      end
    end

    function propUpdatedDynamic(obj,hAx,lObj,propFullName,sPrm,val)
      if obj.initSuccessful && obj.plotOk()
        sPrm_MultiTgt_TargetCrop = sPrm.ROOT.MultiAnimal.IDCropSize;
        assert(startsWith(propFullName,'MultiAnimal.IDCropSize.'));
        toks = strsplit(propFullName,'.');
        propShort = toks{end};
        sPrm_MultiTgt_TargetCrop.(propShort) = val;
        rectPos = obj.getRectPos(lObj,sPrm_MultiTgt_TargetCrop);
        set(obj.hRect,'XData',rectPos(:,1),'YData',rectPos(:,2));
      else
        obj.init(hAx,lObj,propFullName,sPrm);
      end
    end

    function rectPos = getRectPos(obj,lObj,sPrm)
      % rectPos: [5 x 2] col1 is x, col2 is y (closed rectangle for plotting).

      if obj.hasHT && ~isempty(obj.bodyCtrRot)
        % Use the aligned body center and IDCropSize = [x_size, y_size].
        xc = obj.bodyCtrRot(1);
        yc = obj.bodyCtrRot(2);
        id_crop_sz = sPrm;
        half_w = id_crop_sz(1)/2;
        half_h = id_crop_sz(2)/2;
      else
        % No head-tail: fall back to label mean / trx center and square crop.
        if obj.isMA
          xyc = nanmean(obj.xyLbl,1);
          xc = xyc(1);
          yc = xyc(2);
        else
          xc = obj.xTrx;
          yc = obj.yTrx;
        end
        rad = maGetTgtCropRad(sPrm);
        half_w = rad;
        half_h = rad;
      end

      x0 = xc-half_w;
      x1 = xc+half_w;
      y0 = yc-half_h;
      y1 = yc+half_h;
      rectPos = [x0 x0 x1 x1;y0 y1 y1 y0].';

      % for plotting (close the rectangle)
      rectPos(5,:) = rectPos(1,:);
    end

  end

end

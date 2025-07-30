classdef ParameterVisualizationAutoParams < ParameterVisualization
  
  properties
    % If true, a prop for this pvObj is currently selected, and we are 
    % successfully initted/displaying something.
    initSuccessful = false; 
    autoParamsData = struct;
    hHist = [];
    hParams = gobjects(1,0);
    xtick = [];
    xticklabel = {};
    fld = '';
    prmin = [];
  end
  
  properties (Dependent)
    scaledata
    rrangedata
  end

  methods
        
    function init(obj,hTile,lObj,propFullName,prm,vizdata)
      
      if nargin > 1,
        init@ParameterVisualization(obj,hTile,lObj,propFullName,prm);
        obj.autoParamsData = vizdata.autoparams;
        propPath = strsplit(obj.propFullName,'.');
        obj.fld = propPath{end};
        obj.prmin = obj.prm.findnode(obj.propFullName);
      end
      obj.setStage();

      paramdata = [];
      paraminfo = {};
      titles = {};

      if strcmpi(obj.fld,'ManualRadius'),
        data = max(obj.scaledata.bboxWidthHeight,[],1);
        xstr = 'Max bounding box side (pixels)';
        paramdata = obj.prmin.Value;
        paraminfo = {obj.prmin.Data.DisplayNameUse};
      elseif strcmpi(obj.fld,'multi_scale_by_bbox')
        data = obj.scaledata.bboxDiagonalLength;
        xstr = 'Bounding box diagonal (px)';
      elseif strcmpi(obj.fld,'scale_factor_range')
        data = obj.scaledata.bboxDiagonalLength;
        xstr = 'Bounding box diagonal (px)';
        paramdata = obj.scaledata.medBboxDiagonalLength.*[1,1/obj.prmin.Data.Value,obj.prmin.Data.Value];
        paraminfo = {'Median','Median/ScaleFactor', 'Median*ScaleFactor'};
      elseif strcmpi(obj.fld,'rrange'),
        rrangefns = setdiff(fieldnames(obj.rrangedata),{'offset'});
        if numel(rrangefns) == 1,
          rrangefn = rrangefns{1};
        else
          if obj.stage == 1,
            idx = find(startsWith(rrangefns,'firststage'),1);
          else
            idx = find(startsWith(rrangefns,'lasststage'),1);
          end
          if isempty(idx),
            idx = 1;
          end
          rrangefn = rrangefns{idx};
        end
        data = obj.rrangedata.(rrangefn)*180/pi;
        paramdata = modrange(obj.rrangedata.offset.(rrangefn)+[0,[-1,1]*obj.prmin.Data.Value],-180,180);
        paraminfo = {'Median','Median-Rotation','Median+Rotation'};
        switch rrangefn,
          case 'firststage_headTailAngle'
            xstr = 'Tail->Head angle (deg)';
          case 'laststage_keypoints2HeadTailAngle',
            xstr = 'Centroid->Keypoint - Tail->Head angle (deg)';
            titles = obj.lObj.skelNames;
          case 'centroidKeypointAngle',
            xstr = 'Centroid->Keypoint angle (deg)';
          otherwise
            error('Unknown %s',rrangefn);
        end
      else
        error('Unknown field %s',obj.fld);
      end
      binlimits = [min(data(:)),max(data(:))];
      nbins = 20;
      colors = lines(size(paramdata,2));
      obj.hHist = gobjects(1,size(data,1));
      obj.hParams = gobjects(size(data,1),size(paramdata,2));
      for i = 1:size(data,1),
        if (numel(obj.hAx) < i) || ~ishandle(obj.hax(i)),
          obj.hAx(i) = nexttile(obj.hTile);
        end
        obj.hHist(i) = histogram(obj.hAx(i),data(i,:),nbins,'FaceColor','k','BinLimits',binlimits);
        box(obj.hAx(i),'off');
        if numel(titles) >= i,
          title(obj.hAx(i),titles{i});
        end
      end
      linkaxes(obj.hAx);
      ylim = obj.hAx(1).YLim;
      for i = 1:size(data,1),
        hold(obj.hAx(i),'on');
        for j = 1:size(paramdata,2),
          obj.hParams(i,j) = plot(obj.hAx(i),paramdata(i,j)+[0,0],ylim,'-','Color',colors(j,:),'LineWidth',2);
        end
      end
      obj.hAx(end).XTickMode = 'auto';
      obj.hAx(end).XTickLabelMode = 'auto';
      obj.xtick = obj.hAx(end).XTick;
      obj.xticklabel = obj.hAx(end).XTickLabel;
      xtick = obj.xtick; %#ok<*PROPLC>
      xticklabel = obj.xticklabel;
      for i = 1:size(paramdata,j),
        xtick(end+1) = paramdata(end,i); %#ok<AGROW>
        xticklabel{end+1} = paraminfo{end,i}; %#ok<AGROW>
      end
      [xtick,order] = sort(xtick);
      xticklabel = xticklabel(order);
      if ~isempty(paramdata),
        set(obj.hAx,'XTick',xtick);
        set(obj.hAx(1:end-1),'XTickLabel',xticklabel);
        obj.hAx(end).XTickLabel = xticklabel;
      end

      xlabel(obj.hAx(end),xstr);
      ylabel(obj.hAx(end),'N. training examples');

      obj.initSuccessful = true;
    end
    
    function clear(obj)
      clear@ParameterVisualization(obj);
      obj.hAx.XTickMode = 'auto';
      obj.hAx.XTickLabelMode = 'auto';
      obj.initSuccessful = false;
    end

    function update(obj)
      if obj.initSuccessful,

        if strcmpi(obj.fld,'ManualRadius'),
          paramdata = obj.prmin.Data.Value;
          paraminfo = {obj.prmin.Data.DisplayNameUse};
        elseif strcmpi(obj.fld,'multi_scale_by_bbox')
          paramdata = [];
          paraminfo = {};
        elseif strcmpi(obj.fld,'scale_factor_range')
          paramdata = obj.scaledata.medBboxDiagonalLength.*[1,1/prmin.Data.Value,obj.prmin.Data.Value];
          paraminfo = {'Median','Median/ScaleFactor', 'Median*ScaleFactor'};
        else
          error('Unknown field %s',obj.fld);
        end
        xtick = obj.xtick; %#ok<*PROP>
        xticklabel = obj.xticklabel;
        for i = 1:numel(paramdata),
          obj.hParams(i).XData = paramdata(i)+[0,0];
          xtick(end+1) = paamdata(i); %#ok<AGROW>
          xticklabel{end+1} = paraminfo{i}; %#ok<AGROW>
        end
        [xtick,order] = sort(xtick);
        xticklabel = xticklabel(order);
        if ~isempty(paramdata),
          obj.hAx.XTick = xtick;
          obj.hAx.XTickLabel = xticklabel;
        end

      else
        obj.init();
      end
    end

    
  end

  methods
    function s = get.scaledata(obj)
      s = obj.autoParamsData.scale;
    end
    function s = get.rrangedata(obj)
      s = obj.autoParamsData.rrange;
    end
  end
  
end
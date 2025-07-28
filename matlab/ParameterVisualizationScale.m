classdef ParameterVisualizationScale < ParameterVisualization
  
  properties
    % If true, a prop for this pvObj is currently selected, and we are 
    % successfully initted/displaying something.
    initSuccessful = false; 
    scaledata = struct;
    hHist = [];
    hParams = gobjects(1,0);
    xtick = [];
    xticklabel = {};
    fld = '';
    prmin = [];
  end
  
  methods
        
    function init(obj,hTile,lObj,propFullName,prm,vizdata)
      
      if nargin > 1,
        init@ParameterVisualization(obj,hTile,lObj,propFullName,prm);
        obj.scaledata = vizdata.autoparams.scale;
        propPath = strsplit(obj.propFullName,'.');
        obj.fld = propPath{end};
        obj.prmin = obj.prm.findnode(obj.propFullName);
      end

      if isempty(obj.hAx),
        obj.hAx = nexttile(obj.hTile);
      end

      if strcmpi(obj.fld,'ManualRadius'),
        data = max(obj.scaledata.bboxWidthHeight,[],1);
        xstr = 'Max bounding box side (pixels)';
        paramdata = obj.prmin.Value;
        paraminfo = {obj.prmin.Data.DisplayNameUse};
      elseif strcmpi(obj.fld,'multi_scale_by_bbox')
        data = obj.scaledata.bboxDiagonalLength;
        xstr = 'Bounding box diagonal (px)';
        paramdata = [];
        paraminfo = {};
      elseif strcmpi(obj.fld,'scale_factor_range')
        data = obj.scaledata.bboxDiagonalLength;
        xstr = 'Bounding box diagonal length (pixels)';
        paramdata = obj.scaledata.medBboxDiagonalLength.*[1,1/obj.prmin.Data.Value,obj.prmin.Data.Value];
        paraminfo = {'Median','Median / Scale Factor', 'Median * Scale Factor'};
      else
        error('Unknown field %s',obj.fld);
      end
      binlimits = [min(data),max(data)];
      obj.hHist = histogram(obj.hAx,data,'FaceColor','k','BinLimits',binlimits);
      ylim = obj.hAx.YLim;
      hold(obj.hAx,'on');
      colors = lines(numel(paramdata));
      obj.hParams = gobjects(1,numel(paramdata));
      obj.hAx.XTickMode = 'auto';
      obj.hAx.XTickLabelMode = 'auto';
      obj.xtick = obj.hAx.XTick;
      obj.xticklabel = obj.hAx.XTickLabel;
      xtick = obj.xtick; %#ok<*PROPLC>
      xticklabel = obj.xticklabel;
      for i = 1:numel(paramdata),
        obj.hParams(i) = plot(obj.hAx,paramdata(i)+[0,0],ylim,'-','Color',colors(i,:),'LineWidth',2);
        xtick(end+1) = paramdata(i); %#ok<AGROW>
        xticklabel{end+1} = paraminfo{i}; %#ok<AGROW>
      end
      [xtick,order] = sort(xtick);
      xticklabel = xticklabel(order);
      if ~isempty(paramdata),
        obj.hAx.XTick = xtick;
        obj.hAx.XTickLabel = xticklabel;
      end

      xlabel(obj.hAx,xstr);
      ylabel(obj.hAx,'N. training examples');

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
          paraminfo = {'Median','Median / Scale Factor', 'Median * Scale Factor'};
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
  
end
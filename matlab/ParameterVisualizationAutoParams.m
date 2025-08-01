classdef ParameterVisualizationAutoParams < ParameterVisualization
  
  properties
    % If true, a prop for this pvObj is currently selected, and we are 
    % successfully initted/displaying something.
    initSuccessful = false; 
    autoParamsData = struct;
    hHist = [];
    hParams = gobjects(1,0);
    hParamsText = gobjects(1,0);
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

      [data,xstr,paramdata,paraminfo,titles] = obj.getData();

      binlimits = [min(data(:)),max(data(:))];
      nbins = 20;
      colors = lines(size(paramdata,2));
      obj.hHist = gobjects(1,size(data,1));
      obj.hParams = gobjects(size(data,1),size(paramdata,2));
      obj.hParamsText = gobjects(size(data,1),size(paramdata,2));
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
      if size(paramdata,2) == 1,
        yfactor = .9;
      else
        yfactor = linspace(.7,.95,size(paramdata,2));
      end
      for i = 1:size(data,1),
        hold(obj.hAx(i),'on');
        for j = 1:size(paramdata,2),
          obj.hParams(i,j) = plot(obj.hAx(i),paramdata(i,j)+[0,0],ylim,'-','Color',colors(j,:),'LineWidth',2);
          obj.hParamsText(i,j) = text(obj.hAx(i),paramdata(i,j),sum(ylim.*[1-yfactor(j),yfactor(j)]),[' ',paraminfo{j}],'Color',colors(j,:));
        end
      end
      set(obj.hAx(1:end-1),'XTick',[]);
      xlabel(obj.hAx(end),xstr);
      ylabel(obj.hAx(end),'N. training examples');

      obj.initSuccessful = true;
    end
    
    function [data,xstr,paramdata,paraminfo,titles] = getData(obj)

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
        paramdata = obj.scaledata.medBboxDiagonalLength.*[1/obj.prmin.Data.Value,1,obj.prmin.Data.Value];
        paraminfo = {'Median/ScaleFactor', 'Median', 'Median*ScaleFactor'};
      elseif strcmpi(obj.fld,'rrange'),
        rrangefns = setdiff(fieldnames(obj.rrangedata),{'offset'});
        if numel(rrangefns) == 1,
          rrangefn = rrangefns{1};
        else
          if obj.stage == 1,
            idx = find(startsWith(rrangefns,'firststage'),1);
          else
            idx = find(startsWith(rrangefns,'laststage'),1);
          end
          if isempty(idx),
            idx = 1;
          end
          rrangefn = rrangefns{idx};
        end
        data = obj.rrangedata.(rrangefn)*180/pi;
        paramdata = modrange(obj.rrangedata.offset.(rrangefn)*180/pi+[-1,0,1]*obj.prmin.Data.Value,-180,180);
        paraminfo = {'Median-Rotation','Median','Median+Rotation'};
        rrangem = regexp(rrangefn,'^(?<stage>.*stage)?_?(?<name>.*)$','names','once');
        switch rrangem.name,
          case 'firststage_headTailAngle'
            xstr = 'Tail->Head angle (deg)';
          case 'laststage_keypoints2HeadTailAngle',
            xstr = 'Centroid->Keypoint - Tail->Head angle (deg)';
            titles = cellfun(@(s) ['Keypoint ',s],obj.lObj.skelNames,'Uni',0);
          case 'centroidKeypointAngle',
            xstr = 'Centroid->Keypoint angle (deg)';
            titles = cellfun(@(s) ['Keypoint ',s],obj.lObj.skelNames,'Uni',0);
          otherwise
            error('Unknown %s',rrangefn);
        end
      else
        error('Unknown field %s',obj.fld);
      end

    end

    function clear(obj)
      clear@ParameterVisualization(obj);
      obj.initSuccessful = false;
    end

    function update(obj)
      if obj.initSuccessful,

        [~,~,paramdata,~,~] = obj.getData();

        for i = 1:size(paramdata,1),
          for j = 1:size(paramdata,2),
            obj.hParams(i,j).XData = paramdata(i,j)+[0,0];
            pos = obj.hParamsText(i,j).Position;
            pos(1) = paramdata(i,j);
            obj.hParamsText(i,j).Position = pos;
          end
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
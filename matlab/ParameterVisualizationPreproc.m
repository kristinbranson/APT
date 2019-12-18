classdef ParameterVisualizationPreproc < ParameterVisualization
  
  properties
    initSuccessful = false;
    initVizInfo % scalar struct with info for updating plot
    tfUpdating = false;
  end
  
  methods
    
    function propSelected(obj,hAx,lObj,propFullName,sPrm)
      if ~obj.initSuccessful,
        obj.init(hAx,lObj,sPrm,propFullName);
      else
        obj.update(hAx,lObj,sPrm,propFullName);
      end
    end
    
    function propUnselected(obj) %#ok<MANU>
%       obj.initSuccessful = false;
%       obj.initVizInfo = [];
    end

    function propUpdated(obj,hAx,lObj,propFullName,sPrm)
      %prmFtr = sPrm.ROOT.CPR.Feature;
      obj.update(hAx,lObj,sPrm,propFullName);
    end

    function propUpdatedDynamic(obj,hAx,lObj,propFullName,sPrm,val) %#ok<INUSD>
      

      try
        eval(sprintf('sPrm.ROOT.%s;',propFullName));
      catch 
        warningNoTrace(sprintf('Unknown property %s',propFullName));
        return;
      end
      eval(sprintf('sPrm.ROOT.%s = val;',propFullName));
      
%       % to do: store val in sPrm
%       switch propFullName,
%         case 'ImageProcessing.Multiple Targets.Target ROI.Pad background'
%           sPrm.ROOT.ImageProcessing.MultiTarget.TargetCrop.PadBkgd = val;
%         case 'ImageProcessing.Histogram Equalization.Enable'
%           sPrm.ROOT.ImageProcessing.HistEq.Use = val;
%         case 'ImageProcessing.Histogram Equalization.Num frames sample'
%           sPrm.ROOT.ImageProcessing.HistEq.NSampleH0 = val;
%         case 'ImageProcessing.Background Subtraction.Enable',
%           sPrm.ROOT.ImageProcessing.BackSub.Use = val;
%         case 'ImageProcessing.Background Subtraction.Background Type',
%           sPrm.ROOT.ImageProcessing.BackSub.BGType = val;
%         case 'ImageProcessing.Background Subtraction.Background Read Function',
%           sPrm.ROOT.ImageProcessing.BackSub.BGReadFcn = val;
%         otherwise
%           error('Unknown property changed: %s',propFullName);
%       end
      
      obj.update(hAx,lObj,sPrm,propFullName);
      
    end
    
    function init(obj,hAx,lObj,sPrm,propFullName)
      % plot sample processed training images
      % Set .initSuccessful, initVizInfo
      % Subsequent changes to can be handled via update(). This avoids
      % recollecting all training labels.

      obj.initSuccessful = false;
      obj.initVizInfo = [];
      set(hAx,'Units','normalized','Position',obj.axPos);
            
      if ~lObj.hasMovie
        ParameterVisualization.grayOutAxes(hAx,'No movie available.');
        return;
      end
      
      %ParameterVisualization.setBusy(hAx,'Computing visualization. Please wait...');
      
      % Choose labeled frames
      
      nr = 3;
      nc = 3;
      
      obj.initVizInfo = struct;
      obj.initVizInfo.nr = nr;
      obj.initVizInfo.nc = nc;
      obj.initSuccessful = true;

      obj.update(hAx,lObj,sPrm,propFullName);
      

    end
    
    function [tfsucc,msg] = cleanDataAugCache(obj)
      
      tfsucc = true;
      msg = '';
      if ~isfield(obj.initVizInfo,'dataAugDir') || isempty(obj.initVizInfo.dataAugDir),
        return;
      end
      if ~exist(obj.initVizInfo.dataAugDir,'dir'),
        tfsucc = false;
        msg = sprintf('Data aug directory %s does not exist',obj.initVizInfo.dataAugDir);
      else
        fprintf('Removing data aug cache dir %s...\n',obj.initVizInfo.dataAugDir);
        [tfsucc,msg] = rmdir(obj.initVizInfo.dataAugDir,'s');
      end
      obj.initVizInfo.dataAugDir = '';
      obj.initVizInfo.augd = [];
      
    end
    
    function update(obj,hAx,lObj,sPrm,propFullName)

      if obj.tfUpdating,
        %fprintf('Update already running, canceling this call.\n');
        return;
      end
      ParameterVisualization.setBusy(hAx);
      obj.tfUpdating = true;
      
      [~,~,ppPrms] = lObj.convertNew2OldParams(sPrm);
      
      nr = obj.initVizInfo.nr;
      nc = obj.initVizInfo.nc;
      
      if ~isfield(obj.initVizInfo,'tblPTrn1') || isempty(obj.initVizInfo.tblPTrn1) || ...
          ~isfield(obj.initVizInfo,'sPrm') || isempty(obj.initVizInfo.sPrm) || ...
          ~APTParameters.isEqualPreProcParams(obj.initVizInfo.sPrm,sPrm),

        % Let 'treatInfPosAsOcc' default to false here, should be fine as
        % this is for paramviz
        tblPTrn = lObj.preProcGetMFTableLbled('preProcParams',ppPrms);
        nlabeled = size(tblPTrn,1);
        if nr * nc > nlabeled,
          nc = ceil(sqrt(nlabeled));
          nr = ceil(nlabeled/nc);
          obj.initVizInfo.tblPTrn1 = tblPTrn;
        else
          nshow = nr*nc;
          idx = randsample(nlabeled,nshow);
          obj.initVizInfo.tblPTrn1 = tblPTrn(idx,:);
        end
        
        obj.initVizInfo.ppd = lObj.tracker.fetchPreProcData(obj.initVizInfo.tblPTrn1,ppPrms);
        obj.initVizInfo.sPrm = sPrm;
        [tfsucc,msg] = obj.cleanDataAugCache();
        if ~tfsucc,
          warning('Error cleaning data aug cache: %s',msg);
        end

      end
      nshow = size(obj.initVizInfo.tblPTrn1,1);
      
      s = strsplit(propFullName,'.');
      
      % todo: figure out what to do with multiple views, reuse lbl file
      if strcmpi(s{1},'ImageProcessing'),
        ims = obj.initVizInfo.ppd.I;
      elseif strcmpi(s{1},'DeepTrack') && strcmpi(s{2},'DataAugmentation'),
        if ~isfield(obj.initVizInfo,'augd') || isempty(obj.initVizInfo.augd) || ...
            ~APTParameters.isEqualDeepTrackDataAugParams(obj.initVizInfo.sPrm,sPrm),
          fprintf('DATAAUG:\n');
          disp(sPrm.ROOT.DeepTrack.DataAugmentation);

          if isfield(obj.initVizInfo,'dataAugDir'),
            dataAugParams = {'dataAugDir',obj.initVizInfo.dataAugDir};
          else
            dataAugParams = {};
          end
          [obj.initVizInfo.augd,obj.initVizInfo.dataAugDir] = ...
            lObj.tracker.dataAug(obj.initVizInfo.ppd,'sPrmAll',sPrm,dataAugParams{:});
          obj.initVizInfo.sPrm = sPrm;
        end
        nims = cellfun(@(x) size(x,4),obj.initVizInfo.augd.ims);
        ims = cell(sum(nims),1);
        k = 1;
        for i = 1:numel(obj.initVizInfo.augd.ims),
          for j = 1:size(obj.initVizInfo.augd.ims{i},4),
            ims{k} = obj.initVizInfo.augd.ims{i}(:,:,:,j)/sPrm.ROOT.DeepTrack.ImageProcessing.imax;
            k = k + 1;
          end
        end
      else
        obj.tfUpdating = false;
        error('Not implemented');
      end
      
      imsz = cellfun(@size,ims,'Uni',0);
      for i = 1:numel(imsz),
        if numel(imsz{i}) < 3,
          imsz{i}(end+1) = 1;
        end
      end
      maxr = max(cellfun(@(x) x(:,1),imsz));
      maxc = max(cellfun(@(x) x(:,2),imsz));
      maxchn = max(cellfun(@(x) x(:,3),imsz));
      im = zeros([maxr*nr,maxc*nc,maxchn],class(ims{1}));
      toplefts = nan(nshow,2);
      for i = 1:nshow,
        [r,c] = ind2sub([nr,nc],i);
        offr = (r-1)*maxr;
        offc = (c-1)*maxc;
        imcurr = ims{i};
        if imsz{i}(3) < maxchn,
          imcurr = repmat(imcurr(:,:,1),[1,1,maxchn]);
        end
        im(offr+1:offr+imsz{i}(1),offc+1:offc+imsz{i}(2),:) = imcurr;
        toplefts(i,:) = [offr,offc]+1;
      end      
      
      cla(hAx);
      hold(hAx,'off');
      imshow(im,'Parent',hAx);
      hold(hAx,'on');
      for i = 1:nshow,
        text(toplefts(i,1),toplefts(i,2),...
          sprintf('mov %d, tgt %d, frm %d',...
          obj.initVizInfo.tblPTrn1.mov(i),obj.initVizInfo.tblPTrn1.iTgt(i),...
          obj.initVizInfo.tblPTrn1.frm(i)),...
          'HorizontalAlignment','left','VerticalAlignment','top','Color','c',...
          'Parent',hAx);
      end
      hAx.XTick = [];
      hAx.YTick = [];
      
      obj.tfUpdating = false;
      ParameterVisualization.setReady(hAx);
      
    end
     
    function delete(obj)
      
      if isstruct(obj.initVizInfo),
        try
          obj.cleanDataAugCache();
        catch ME
          warning(getReport(ME));
        end
      end
      obj.initVizInfo = [];
      
      
    end
    
  end
  
end
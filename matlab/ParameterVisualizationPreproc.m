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

    function propUpdatedDynamic(obj,hAx,lObj,propFullName,prm,val) %#ok<INUSD>
      
      propFullName = ParameterVisualizationPreproc.ParameterVisualizationPreproc.modernizePropName(propFullName);
      try
        ParameterVisualizationPreproc.getParamValue(prm,propFullName);
      catch 
        warningNoTrace(sprintf('Unknown property %s',propFullName));
        return;
      end
      if isstruct(prm),
        eval(sprintf('prm.%s = val;',propFullName));
      end
      
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
      
      obj.update(hAx,lObj,prm,propFullName);
      
    end
    
    function init(obj,hAx,lObj,prm,propFullName)
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
      
      [tffound] = lObj.labelFindOneLabeledFrame();
      if ~tffound,
        ParameterVisualization.grayOutAxes(hAx,'No frames labeled.');
        return;
      end
        
      
      %ParameterVisualization.setBusy(hAx,'Computing visualization. Please wait...');
      
      obj.initializeVizInfo();
      
      obj.update(hAx,lObj,prm,propFullName);
      

    end
    
    function initializeVizInfo(obj)
      nr = 3;
      nc = 3;
      
      obj.initVizInfo = struct;
      obj.initVizInfo.nr = nr;
      obj.initVizInfo.nc = nc;
      obj.initSuccessful = true;

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
      if ~isstruct(sPrm),
        sPrm = sPrm.structize();
      end

      if obj.tfUpdating,
        %fprintf('Update already running, canceling this call.\n');
        return;
      end
      ParameterVisualization.setBusy(hAx);
      obj.tfUpdating = true;
      
%       [~,~,ppPrms] = lObj.convertNew2OldParams(sPrm);
      
      if ~isstruct(obj.initVizInfo),
        obj.initializeVizInfo();
      end
      
      nr = obj.initVizInfo.nr;
      nc = obj.initVizInfo.nc;
      
      if ~isfield(obj.initVizInfo,'tblPTrn1') || isempty(obj.initVizInfo.tblPTrn1) || ...
          ~isfield(obj.initVizInfo,'sPrm') || isempty(obj.initVizInfo.sPrm) || ...
          ~APTParameters.isEqualPreProcParams(obj.initVizInfo.sPrm,sPrm),

        % Let 'treatInfPosAsOcc' default to false here, should be fine as
        % this is for paramviz
        maTgtCropRad = APTParameters.maGetTgtCropRad(sPrm);
        tblPTrn = lObj.preProcGetMFTableLbled('maTgtCropRad',maTgtCropRad);
        nlabeled = size(tblPTrn,1);
        if nlabeled == 0,
          return;
        end
        if nr * nc > nlabeled,
          nc = ceil(sqrt(nlabeled));
          nr = ceil(nlabeled/nc);
          obj.initVizInfo.tblPTrn1 = tblPTrn;
        else
          nshow = nr*nc;
          idx = randsample(nlabeled,nshow);
          obj.initVizInfo.tblPTrn1 = tblPTrn(idx,:);
        end
        
        obj.initVizInfo.ppd = lObj.tracker.fetchPreProcData(...
          obj.initVizInfo.tblPTrn1,sPrmTgtCrop);
        obj.initVizInfo.sPrm = sPrm;
        [tfsucc,msg] = obj.cleanDataAugCache();
        if ~tfsucc,
          warning('Error cleaning data aug cache: %s',msg);
        end

      end
      nshow = size(obj.initVizInfo.tblPTrn1,1);
      
      s = strsplit(propFullName,'.');
      nparts = lObj.nPhysPoints;
      nview = lObj.nview;
      
      % todo: figure out what to do with multiple views, reuse lbl file
      if strcmpi(s{1},'ImageProcessing'),
        ims = obj.initVizInfo.ppd.I;
        locs = permute(reshape(obj.initVizInfo.ppd.pGT,[numel(ims),nparts,2]),[2,3,1]);
      elseif (strcmpi(s{1},'DeepTrack') || strcmpi(s{1},'Deep Learning (pose)')) && (strcmpi(s{2},'DataAugmentation') || strcmpi(s{2},'Data Augmentation')),
        if ~isfield(obj.initVizInfo,'augd') || isempty(obj.initVizInfo.augd) || ...
            ~APTParameters.isEqualDeepTrackDataAugParams(obj.initVizInfo.sPrm,sPrm),
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
        assert(all(nims==nims(1)));
        nims = nims(1);
        ims = cell(nims,nview);
        locs = nan([nparts,2,nims,nview]);
        imax = APTParameters.getImageProcessingIMax(sPrm);
        for i = 1:nview,
          for j = 1:nims,
            ims{j,i} = obj.initVizInfo.augd.ims{i}(:,:,:,j)/imax;
            locs(:,:,j,i) = permute(obj.initVizInfo.augd.locs{i}(j,:,:),[2,3,1]);
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
      maxr = max(max(cellfun(@(x) x(:,1),imsz)));
      maxchn = max(max(cellfun(@(x) x(:,3),imsz)));
      toplefts = nan([nshow,2,nview]);
      maxc = max(cellfun(@(x) x(:,2),imsz));
      im = zeros([maxr*nr,sum(maxc*nc),maxchn],class(ims{1}));
      for vwi = 1:nview,
        offvw = sum(maxc(1:vwi-1)*nc);
        for i = 1:nshow,
          [r,c] = ind2sub([nr,nc],i);
          offr = (r-1)*maxr;
          offc = offvw+(c-1)*maxc(vwi);
          imcurr = ims{i,vwi};
          if imsz{i,vwi}(3) < maxchn,
            imcurr = repmat(imcurr(:,:,1),[1,1,maxchn]);
          end
          im(offr+1:offr+imsz{i,vwi}(1),offc+1:offc+imsz{i,vwi}(2),:) = imcurr;
          toplefts(i,:,vwi) = [offr,offc]+1;
        end
      end
      
      cla(hAx);
      hold(hAx,'off');
      imshow(im,'Parent',hAx);
      hold(hAx,'on');
      for i = 1:nshow,
        if lObj.hasTrx,
          targs = sprintf('tgt %d, ',obj.initVizInfo.tblPTrn1.iTgt(i));
        else
          targs = '';
        end
        for j = 1:nview,
          text(toplefts(i,2,j),toplefts(i,1,j),...
            sprintf('mov %d, %sfrm %d',...
            obj.initVizInfo.tblPTrn1.mov(i),targs,...
            obj.initVizInfo.tblPTrn1.frm(i)),...
            'HorizontalAlignment','left','VerticalAlignment','top','Color','c',...
            'Parent',hAx);
        end
      end
      
      for i = 1:nparts,
        for vwi = 1:nview,
          plot(hAx,squeeze(locs(i,1,:,vwi))+toplefts(:,2,vwi)-1,squeeze(locs(i,2,:,vwi))+toplefts(:,1,vwi)-1,...
          '.','Color',lObj.labelPointsPlotInfo.Colors(i,:));
        end
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
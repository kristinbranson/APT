classdef ParameterVisualizationCPRInit < ParameterVisualization
  
  properties
    initSuccessful = false;
    initVizInfo % scalar struct with info for updating plot
  end
  
  methods
    
    function propSelected(obj,hAx,lObj,propFullName,sPrm)
      obj.init(hAx,lObj,sPrm);
    end
    
    function propUnselected(obj)
      obj.initSuccessful = false;
      obj.initVizInfo = [];
    end

    function propUpdated(obj,hAx,lObj,propFullName,sPrm)
      %prmFtr = sPrm.ROOT.CPR.Feature;
      obj.init(hAx,lObj,sPrm);
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
      
      obj.init(hAx,lObj,sPrm);
      
    end
    
    function init(obj,hAx,lObj,sPrm)
      % plot sample processed training images
      % Set .initSuccessful, initVizInfo
      % Subsequent changes to can be handled via update(). This avoids
      % recollecting all training labels.

      obj.initSuccessful = false;
      obj.initVizInfo = [];
      nrepplot = 10;
            
      if ~lObj.hasMovie
        ParameterVisualization.grayOutAxes(hAx,'No movie available.');
        return;
      end
      
      ParameterVisualization.setBusy(hAx,'Computing visualization. Please wait...');
      
      % Choose labeled frames to read in
      [~,sPrmCPRold,ppPrms] = lObj.convertNew2OldParams(sPrm);
      
      tblPTrn = lObj.preProcGetMFTableLbled('preProcParams',ppPrms);      
      nr = 3;
      nc = 3;
      nlabeled = size(tblPTrn,1);
      if nr * nc > nlabeled,
        nc = ceil(sqrt(nlabeled));
        nr = ceil(nlabeled/nc);
        tblPTrn1 = tblPTrn;
        idx = 1:nlabeled;
      else
        nshow = nr*nc;
        %idx = sort(randsample(nlabeled,nshow));
        idx = unique(round(linspace(1,nlabeled,nshow)));
        tblPTrn1 = tblPTrn(idx,:);
      end
      [~,~,d] = lObj.tracker.preretrain(tblPTrn1,[],ppPrms);
      
      % get initializations
      bboxes = repmat(d.bboxesTrn(1,:),[nlabeled,1]);
      bboxes(idx,:) = d.bboxesTrn;
      p0 = lObj.tracker.randInit(tblPTrn,bboxes,'CPRParams',sPrmCPRold);
      npts = sPrmCPRold.Model.nfids;
      sz = size(p0);
      p0 = reshape(p0,[sz(1:2),npts,2,sz(4:end)]);
      p0 = p0(idx,:,:,:,:);
      

%       obj.initVizInfo = struct;
%       obj.initVizInfo.tblPTrn = tblPTrn;
%       obj.initVizInfo.tblPTrn1 = tblPTrn1;
%       obj.initVizInfo.nr = nr;
%       obj.initVizInfo.nc = nc;
%       obj.initVizInfo.p0 = p0;
     
      % make sure that this is the same as if we use the whole data set,
      % histogram equalization might depend on all the data
      
      imsz = cellfun(@size,d.ITrn,'Uni',0);
      for i = 1:numel(imsz),
        if numel(imsz{i}) < 3,
          imsz{i}(end+1) = 1;
        end
      end
      maxr = max(cellfun(@(x) x(:,1),imsz));
      maxc = max(cellfun(@(x) x(:,2),imsz));
      maxchn = max(cellfun(@(x) x(:,3),imsz));
      im = zeros([maxr*nr,maxc*nc,maxchn],class(d.ITrn{1}));
      toplefts = nan(nshow,2);
      for i = 1:nshow,
        [r,c] = ind2sub([nr,nc],i);
        offr = (r-1)*maxr;
        offc = (c-1)*maxc;
        imcurr = d.ITrn{i};
        if imsz{i}(3) < maxchn,
          imcurr = repmat(imcurr(:,:,1),[1,1,maxchn]);
        end
        im(offr+1:offr+imsz{i}(1),offc+1:offc+imsz{i}(2),:) = imcurr;
        toplefts(i,:) = [offc,offr]+1;
      end      
      
      cla(hAx);
      hold(hAx,'off');
      imshow(im,'Parent',hAx);
      hold(hAx,'on');
      
      colors = lObj.LabelPointColors;
      for i = 1:nshow,
        text(toplefts(i,1),toplefts(i,2),...
          sprintf('mov %d, tgt %d, frm %d',...
          tblPTrn1.mov(i),tblPTrn1.iTgt(i),tblPTrn1.frm(i)),...
          'HorizontalAlignment','left','VerticalAlignment','top','Color','c',...
          'Parent',hAx);
        if nrepplot >= size(p0,2),
          repplot = 1:size(p0,2);
        else
          repplot = randsample(size(p0,2),nrepplot);
        end
        for j = 1:npts,
          plot(toplefts(i,1)+p0(i,repplot,j,1,1)-1,toplefts(i,2)+p0(i,repplot,j,2,1)-1,'.','Color',colors(j,:));
        end
      end
      
      ParameterVisualization.setReady(hAx);
      obj.initSuccessful = true;
      
    end
        
  end
  
end
function sagg = aggregateLabelsAddRoi(lObj,isObjDet,sPrmBBox,...
    sPrmLossMask,varargin)

  [incPartialRows,treatInfPosAsOcc] = myparse(varargin,...
    'incPartialRows',false,...
    'treatInfPosAsOcc',true ...
    );

  isgt = lObj.gtIsGTMode;
  PROPS = lObj.gtGetSharedProps;
  fLbl = PROPS.LBL;
  fmfaf = PROPS.MFAF;

  lbls = lObj.(fLbl);
  mfafs = lObj.(fmfaf);
  nmov = numel(lbls);
  sagg = cell(nmov,1);
  for imov=1:nmov
    s = lbls{imov};
    s.mov = mfafs{imov};

    % see also from Labeler/preProcGetMFTableLbled
    if ~incPartialRows
      s = Labels.rmRows(s,@isnan,'partially-labeled');
    end
    if treatInfPosAsOcc
      s = Labels.replaceInfWithNan(s);
    end

    %% gen rois, bw
    n = size(s.p,2);
    s.roi = nan(8,n);
%        fprintf(1,'mov %d: %d labeled frms.\n',imov,n);
    for i=1:n
      p = s.p(:,i);
      xy = Shape.vec2xy(p);
      if isObjDet
        minaa = sPrmBBox.MinAspectRatio;
        roi = lObj.maComputeBboxGeneral(xy,minaa,false,[],[]);
      else
        roi = lObj.maGetLossMask(xy,sPrmLossMask);
      end
      s.roi(:,i) = roi(:);
    end

    if ~isgt
      sroi = lObj.labelsRoi{imov};
    else
      sroi = LabelROI.new();
    end
    s.frmroi = sroi.f;
    s.extra_roi = sroi.verts;
    sagg{imov} = s;
  end
  sagg = cell2mat(sagg);
end % function

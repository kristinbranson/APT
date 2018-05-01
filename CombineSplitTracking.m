function info = CombineSplitTracking(lblFile,infoFile,outTrkFile)

[info,isdone] = CheckSplitTracking(lblFile,infoFile);

if ~all(isdone),
  return;
end

fprintf('All jobs are complete! Combining!\n');
for i = 1:numel(info),
  fprintf('Loading trk file %d / %d\n',i,numel(info));
  tdcurr = load(info(i).trkFile,'-mat');
  nFramesCurr = info(i).endFrame-info(i).startFrame+1;
  fractrked = nnz(sum(sum(sum(~isnan(tdcurr.pTrk(:,:,info(i).startFrame:info(i).endFrame,:)),1),2),4)>0) / nFramesCurr;
  if fractrked < 1,
    warning('Job %d: %d / %d frames have no tracking data',i,(1-fractrked)*nFramesCurr,nFramesCurr);
  end
  if i == 1,
    td = tdcurr;
  else
    fracprevtrked = nnz(sum(sum(sum(~isnan(td.pTrk(:,:,info(i).startFrame:info(i).endFrame,:)),1),2),4)>0) / nFramesCurr;
    if fracprevtrked > 0,
      warning('Job %d: %d / %d frames had been tracked in a previous job, overwriting.',i,fracprevtrked*nFramesCurr,nFramesCurr);
    end
    td.pTrk(:,:,info(i).startFrame:info(i).endFrame,:) = tdcurr.pTrk(:,:,info(i).startFrame:info(i).endFrame,:);
    td.pTrkTS(:,info(i).startFrame:info(i).endFrame,:) = tdcurr.pTrkTS(:,info(i).startFrame:info(i).endFrame,:);
    td.pTrkTag(:,info(i).startFrame:info(i).endFrame,:) = tdcurr.pTrkTag(:,info(i).startFrame:info(i).endFrame,:);
    if ~isempty(tdcurr.pTrkFullFT),
      [td.pTrkFullFT,idxcombine] = unique([tdcurr.pTrkFullFt;td.pTrkFullFT]);
      isnew = idxcombine<= size(tdcurr.pTrkFullFt,1);
      sz = size(tdcurr.pTrkFull);
      
      td.pTrkFull = nan([sz(1:3),size(td.pTrkFullFt,1),sz(5:end)]);
      td.pTrkFull(:,:,:,isnew,:) = tdcurr.pTrkFull(:,:,:,idxcombine(isnew),:);
      td.pTrkFull(:,:,:,~isnew,:) = td.pTrkFull(:,:,:,idxcombine(~isnew)-size(tdcurr.pTrkFullFT,1),:);
    end
    
  
  end
end
isTracked = sum(sum(sum(~isnan(td.pTrk),1),2),4)>0;
fprintf('Saving tracking for %d frames between %d and %d to %s...\n',nnz(isTracked),...
  find(isTracked,1,'first'),find(isTracked,1,'last'),outTrkFile);
save(outTrkFile,'-struct','td');
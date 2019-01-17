function info = CombineSplitTracking(lblFile,infoFile,outTrkFile,varargin)

[logfile,docheckoverlap] = myparse(varargin,'logfile','','docheckoverlap',true);

islogfile = ~isempty(logfile);
if ~islogfile,
  fid = 1;
else
  fid = fopen(logfile,'w');
end
  
[info,isdone] = CheckSplitTracking(lblFile,infoFile);

if ~all(isdone),
  return;
end

fprintf(fid,'All jobs are complete! Combining!\n');
for i = 1:numel(info),
  fprintf(fid,'Loading trk file %d / %d\n',i,numel(info));
  tdcurr = load(info(i).trkFile,'-mat');
  nFramesCurr = info(i).endFrame-info(i).startFrame+1;
  fractrked = nnz(sum(sum(sum(~isnan(tdcurr.pTrk(:,:,info(i).startFrame:info(i).endFrame,:)),1),2),4)>0) / nFramesCurr;
  if fractrked < 1,
    warning('Job %d: %d / %d frames have no tracking data',i,(1-fractrked)*nFramesCurr,nFramesCurr);
    if islogfile,
      fprintf(fid,'Job %d: %d / %d frames have no tracking data\n',i,(1-fractrked)*nFramesCurr,nFramesCurr);
    end
  end
  if i == 1,
    td = tdcurr;
    td.pTrkFull = [];
    td.pTrkFullFT = [];
%     pTrkFull = td.pTrkFull;
%     pTrkFullFT = td.pTrkFullFT;
  else
    if docheckoverlap,
      fracprevtrked = nnz(sum(sum(sum(~isnan(td.pTrk(:,:,info(i).startFrame:info(i).endFrame,:)),1),2),4)>0) / nFramesCurr;
      if fracprevtrked > 0,
        warning('Job %d: %d / %d frames had been tracked in a previous job, overwriting.',i,fracprevtrked*nFramesCurr,nFramesCurr);
        if islogfile,
          fprintf(fd,'Job %d: %d / %d frames had been tracked in a previous job, overwriting.\n',i,fracprevtrked*nFramesCurr,nFramesCurr);
        end
      end
    end
    td.pTrk(:,:,info(i).startFrame:info(i).endFrame,:) = tdcurr.pTrk(:,:,info(i).startFrame:info(i).endFrame,:);
    td.pTrkTS(:,info(i).startFrame:info(i).endFrame,:) = tdcurr.pTrkTS(:,info(i).startFrame:info(i).endFrame,:);
    td.pTrkTag(:,info(i).startFrame:info(i).endFrame,:) = tdcurr.pTrkTag(:,info(i).startFrame:info(i).endFrame,:);
  end
  if ~isempty(tdcurr.pTrkFullFT),
    if isempty(td.pTrkFull),
      nReps = size(tdcurr.pTrkFull,3);
      [npts,d,Tcurr,nTargets] = size(td.pTrk);
      td.pTrkFull = nan([npts*d*nReps,Tcurr,nTargets],'single');
    end
    for iTgt = 1:nTargets,
      idxcurr = find(tdcurr.pTrkFullFT.iTgt==iTgt);
      if isempty(idxcurr),
        continue;
      end
      % assuming frames are in order
      t0 = tdcurr.pTrkFullFT.frm(idxcurr(1));
      t1 = tdcurr.pTrkFullFT.frm(idxcurr(end));
      assert(all((t0:t1)==tdcurr.pTrkFullFT.frm(idxcurr)'));
      assert(t0>=info(i).startFrame && t1 <= info(i).endFrame);
      td.pTrkFull(:,t0:t1,iTgt) = reshape(tdcurr.pTrkFull(:,:,:,idxcurr),[npts*d*nReps,t1-t0+1]);
    end
    %         pTrkFull(:,:,:,end+1:end+size(tdcurr.pTrkFull,4),:) = tdcurr.pTrkFull(:,:,:,:,:);
    %         pTrkFullFT(end+1:end+size(tdcurr.pTrkFullFT,1),:) = tdcurr.pTrkFullFT;
    %       [td.pTrkFullFT,idxcombine] = unique([tdcurr.pTrkFullFT;td.pTrkFullFT]);
    %       isnew = idxcombine<= size(tdcurr.pTrkFullFT,1);
    %       sz = size(tdcurr.pTrkFull);
    %
    %       pTrkFull = nan([sz(1:3),size(td.pTrkFullFT,1),sz(5:end)]);
    %       pTrkFull(:,:,:,isnew,:) = tdcurr.pTrkFull(:,:,:,idxcombine(isnew),:);
    %       pTrkFull(:,:,:,~isnew,:) = td.pTrkFull(:,:,:,idxcombine(~isnew)-size(tdcurr.pTrkFullFT,1),:);
    %       td.pTrkFull = pTrkFull;
  end
  
  
  if islogfile,
    fclose(fid);
    fid = fopen(logfile,'a');
  end

end

if ~isempty(td.pTrkFull),
  [~,Tcurr,nTargets] = size(td.pTrkFull);
  td.pTrkFull = reshape(td.pTrkFull,[npts,d,nReps,Tcurr,nTargets]);
%   [td.pTrkFullFT,idx] = unique(pTrkFullFT);
%   if numel(idx) == size(pTrkFullFT,1),
%     td.pTrkFullFT = pTrkFullFT;
%     td.pTrkFull = pTrkFull;
%   else
%     sz = size(pTrkFull);
%     td.pTrkFull = reshape(pTrkFull(:,:,:,idx,:),[sz(1:3),numel(idx),sz(5:end)]);
%   end
end


isTracked = sum(sum(sum(~isnan(td.pTrk),1),2),4)>0;
fprintf(fid,'Saving tracking for %d frames between %d and %d to %s...\n',nnz(isTracked),...
  find(isTracked,1,'first'),find(isTracked,1,'last'),outTrkFile);
if islogfile,
  fclose(fid);
end
save(outTrkFile,'-struct','td');

function tbl = kinematCompileSH(kineList)

s = struct(...
  'lblCat',cell(0,1),... % int enum for type of label file
  'lblFile',cell(0,1),... % 3-level path
  'iMov',[],... % movie index within lblFile
  'movFile',[],... % [1xnview] movie fullpaths
  'flyID',[],...
  'frm',[],... 
  'pLbl',[],... % [1xnLabelPoints*2]==[1xnphyspts*nvw*2]
  'pLblTSmin',[],... % scalar, minimum label timestamp 
  'pLblTSmax',[]); % scalar, max label timestamp 
  
nLbl = numel(kineList);
for ilbl=1:nLbl
  disp(ilbl);
  kFile = kineList{ilbl};
  kd = load(kFile,'-mat');
  kd = kd.data;
  lblCat = 0;
  nMov = 1;
  kFile3 = FSPath.threeLevelFilename(kFile);
  
  nphyspts = kd.cfg.NumLabelPoints;  
  nviews = kd.cfg.NumViews;
  fprintf(1,'lbl: %s. %d physpts %d vws %d movs.\n',kFile3,nphyspts,nviews,nMov);
  
  for iMov=1:nMov
    lpos = kd.labeledpos{iMov};
    lposTS = kd.labeledposTS{iMov};
    fLbled = frameLabeled(lpos);
    nf = numel(fLbled);
    if nf>0
      movFiles = kd.movieFilesAll(iMov,:);
      movVw1 = kd.movieFilesAll{iMov,1};
      flyid = parseSHfullmovie(movVw1);
      
      switch nphyspts
        case 5
          pLbl = lpos(:,:,fLbled,:);
          pLblTS = lposTS(:,fLbled);
        case 10
          IPT = [1:5 11:15];
          pLbl = lpos(IPT,:,fLbled,:); % pts 1-5 in views 1-2
          pLblTS = lposTS(IPT,fLbled);
        otherwise
          assert(false);
      end
      szassert(pLbl,[10 2 nf]);
      pLbl = reshape(pLbl,[20 nf])';
      szassert(pLblTS,[10 nf]);
      pLblTSmin = min(pLblTS,[],1)'; % [nf 1]
      pLblTSmax = max(pLblTS,[],1)'; % [nf 1]
                
%       tsmin = min(ts(:));
%       tsmax = max(ts(:));
%       fprintf(fh,'  mov%02d. fly %d. %d lbled frames. min/max ts: %s/%s.\n',...
%         imov,flyid,numel(fLbled),...
%         datestr(tsmin,'yyyymmdd'),datestr(tsmax,'yyyymmdd'));
      
%       lpos = lpos(:,:,fLbled,:);
%       szassert(lpos,[
    
      for iF=1:nf
        s(end+1,1).lblCat = lblCat;
        s(end).lblFile = FSPath.threeLevelFilename(kFile);
        s(end).iMov = iMov;
        s(end).movFile = movFiles;
        s(end).flyID = flyid;
        s(end).frm = fLbled(iF);
        s(end).pLbl = pLbl(iF,:);
        s(end).pLblTSmin = pLblTSmin(iF);
        s(end).pLblTSmax = pLblTSmax(iF);
      end      
    end
  end
end

tbl = struct2table(s);

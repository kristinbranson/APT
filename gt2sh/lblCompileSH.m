function tbl = lblCompileSH(lblList)
% tbl = lblCompileSH(lblList)
% Compile labeled frame info from a list of lbl files
%

s = struct(...
  'lblCat',cell(0,1),... % int enum for type of label file
  'lblFile',cell(0,1),... % 3-level path
  'iMov',[],... % movie index within lblFile
  'movFile',[],... % [1xnview] movie fullpaths
  'movID',[],...
  'movID2',[],...
  'flyID',[],...
  'frm',[],... 
  'pLbl',[],... % [1xnLabelPoints*2]==[1xnphyspts*nvw*2]
  'pLblTSmin',[],... % scalar, minimum label timestamp 
  'pLblTSmax',[]); % scalar, max label timestamp 
  
nLbl = numel(lblList);
for ilbl=1:nLbl
  disp(ilbl);
  lblFile = lblList{ilbl};
  lbl = load(lblFile,'-mat');
  [~,lblCat] = shmovieCat(lblFile);
  nMov = size(lbl.movieFilesAll,1);
  lblFile3 = FSPath.nLevelFnameChar(lblFile,3);
  
  nphyspts = lbl.cfg.NumLabelPoints;  
  nviews = lbl.cfg.NumViews;
  fprintf(1,'lbl: %s. %d physpts %d vws %d movs.\n',lblFile3,nphyspts,nviews,nMov);
  
  for iMov=1:nMov
    lpos = lbl.labeledpos{iMov};
    lposTS = lbl.labeledposTS{iMov};
    if isstruct(lpos)
      lpos = SparseLabelArray.full(lpos);
    end
    if isstruct(lposTS)
      lposTS = SparseLabelArray.full(lposTS);
    end
     
    fLbled = frameLabeled(lpos);
    nf = numel(fLbled);
    if nf>0
      movFiles = lbl.movieFilesAll(iMov,:);
      [flyid,movID] = parseSHfullmovie(movFiles{1});
      [flyid2,movID2] = parseSHfullmovie(movFiles{2});
      assert(flyid==flyid2);      
      
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
        s(end).lblFile = lblFile3;
        s(end).iMov = iMov;
        s(end).movFile = movFiles;
        s(end).movID = movID;
        s(end).movID2 = movID2;
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

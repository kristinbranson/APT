classdef MFTable
  % MFTable -- Movie/Frame tables
  %
  % An MFTable is a table with cols 'mov' and 'frm' indicating movies and
  % frame numbers. 
  %
  % 'mov' is usually filled with "movieIDs": FSPath/"standardized" 
  % moviepaths, which can include macros. For multiview data, 'mov' can 
  % contain multiuple movieIDs delimited by #.
  
  properties (Constant)
    % Uniquely IDs a frame/target
    FLDSID = {'mov' 'frm' 'iTgt'};
    
    % Core training/test data. Notion of roi (for multitarget) has been
    % abstracted away
    FLDSCORE = {'mov' 'frm' 'iTgt' 'tfocc' 'p'};

    FLDSFULL = {'mov' 'frm' 'iTgt' 'p' 'pTS' 'tfocc'};
    
    %  mov    single string unique ID for movieSetID combo
    %  frm    1-based frame index
    %  iTgt   1-based trx index
    %  p      Absolute label positions (px). Raster order: physpt,view,{x,y}
    %  pTS    [npts=nphyspt*nview] timestamps
    %  tfocc  [npts=nphyspt*nview] logical occluded flag
    %  pTrx   [nview*2], trx .x and .y. Raster order: view,{x,y}
    FLDSFULLTRX = {'mov' 'frm' 'iTgt' 'p' 'pTS' 'tfocc' 'pTrx'};    
  end
  
  methods (Static)
    
    function [tblPnew,tblPupdate,idx0update] = tblPDiff(tblP0,tblP)
      % Compare tblP to tblP0 wrt fields FLDSCORE (see below)
      %
      % tblP0, tblP: MF tables
      %
      % tblPNew: new frames (rows of tblP whose movie-frame-tgt ID are not in tblP0)
      % tblPupdate: existing frames with new positions/tags (rows of tblP
      %   whos movie-frame-tgt ID are in tblP0, but whose eg p field is different).
      % idx0update: indices into rows of tblP0 corresponding to tblPupdate;
      %   ie tblP0(idx0update,:) ~ tblPupdate
      
      FLDSID = MFTable.FLDSID;
      FLDSCORE = MFTable.FLDSCORE;
      tblfldscontainsassert(tblP0,FLDSCORE);
      tblfldscontainsassert(tblP,FLDSCORE);
      
      tfNew = ~ismember(tblP(:,FLDSID),tblP0(:,FLDSID));
      tfDiff = ~ismember(tblP(:,FLDSCORE),tblP0(:,FLDSCORE));
      tfUpdate = tfDiff & ~tfNew; 
            
      tblPnew = tblP(tfNew,:);
      tblPupdate = tblP(tfUpdate,:);            
     
      [tf,loc] = ismember(tblPupdate(:,FLDSID),tblP0(:,FLDSID));
      assert(all(tf));
      idx0update = loc;
    end

    function movID = formMultiMovieID(movs)
      % Form multimovie char ID
      %
      % movs: row cellstr vec
      %
      % movID: char
      
      assert(iscellstr(movs) && isrow(movs));
      movID = sprintf('%s#',movs{:});
      movID = movID(1:end-1);
    end
    
    function movs = unpackMultiMovieID(movID)
      movs = regexp(movID,'#','split');
    end
    
    function I = fetchImages(tMF)
      %
      % tMF: MFTable, n rows.
      %
      % I: [nxnview]
      
      movsets = tMF.movSet;
      movIDs = cellfun(@MFTable.formMultiMovieID,movsets,'uni',0);
      [movIDsUn,idx] = unique(movIDs);
      
      % open moviereaders
      movsetsUn = movsets(idx);
      movsetsUn = cat(1,movsetsUn{:});
      [nMovsetsUn,nView] = size(movsetsUn);
      mrcell = cell(size(movsetsUn));
      for iMovSet=1:nMovsetsUn
        for iView=1:nView
          mr = MovieReader();
          mr.open(movsetsUn{iMovSet,iView});
          mr.forceGrayscale = true;
          mrcell{iMovSet,iView} = mr;
        end
      end
      
      nRows = size(tMF,1);
      I = cell(nRows,nView);
      for iRow=1:nRows
        frm = tMF.frm(iRow);
        id = movIDs{iRow};
        iMovSet = strcmp(id,movIDsUn);
        assert(nnz(iMovSet)==1);
        for iView=1:nView
          I{iRow,iView} = mrcell{iMovSet,iView}.readframe(frm);
        end
        if mod(iRow,10)==0
          fprintf(1,'Read images: row %d\n',iRow);
        end
      end
    end
    
    function tblMF = rmMovS(tblMF)
      % remove legacy 'movS' field from MFTable
      
      if any(strcmp(tblMF.Properties.VariableNames,'movS'))
        [~,movSexpected] = cellfun(@myfileparts,tblMF.mov,'uni',0);
        tf = ~strcmp(movSexpected,tblMF.movS);
        if any(tf)
          warnstr = strcat(movSexpected(tf),{' vs. '},tblMF.movS(tf));
          warnstr = unique(warnstr);
          cellfun(@(x)warningNoTrace('MFTable:movS','Unexpected movS field: %s.',x),warnstr);
        end
        tblMF(:,'movS') = [];
      end
    end
    
    function warnDupMovFrmKey(tblMF,tblDescStr)
      % Warn when tblMF contains duplicate mov|frm IDs
      
      tMF = tblMF(:,MFTable.FLDSID);
      sz0 = size(tMF,1);
      tMF = unique(tMF);
      sz1 = size(tMF,1);
      if sz0~=sz1
        ndup = sz0-sz1;
        warnstr = sprintf('%s table contains %d duplicate movie/frame keys. This should not happen and could indicate a data corruption.',...
          tblDescStr,ndup);
        warndlg(warnstr,'Duplicate data keys found','modal');        
      end
    end
    
    function tblMF = replaceMovieFullWithMovieID(tblMF,movIDs,movFull,tblDesc)
      % Replace "platformized"/full movies with movieIDs/keys
      
      szassert(movIDs,size(movFull));
      [tf,loc] = ismember(tblMF.mov,movFull);
      tblMF.mov(tf) = movIDs(loc(tf));
      replacestrs = strcat(movFull(loc(tf)),{'->'},movIDs(loc(tf)));
      replacestrs = unique(replacestrs);
      cellfun(@(x)warningNoTrace('MFTable:mov',...
        '%s table: replaced %s to match project.',tblDesc,x),replacestrs);
    end
          
  end
  
end
  
  
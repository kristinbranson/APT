classdef MFTable
% Movie/Frame/(Target) tables
%
% An MFTable is a table with cols 'mov' and 'frm' indicating movies and
% frame numbers.

  methods (Static) % General table utils
    
    function [tblNew,tfRm] = remapIntegerKey(tbl,keyfld,keymap)
      % tbl: any table with a "key" field which takes nonzero integer
      %   values; MovieIndex values also accepted
      % keyfld: fieldname of tbl
      % map: containers.Map where map(oldKeyVal)==newKeyVal. if
      %   map(oldKeyVal)==0, then that row is to be removed.
      %
      % tblNew: tbl with keys remapped and rows removed as appropriate
      % tfRm: [height(tbl)x1] logical. True for rows of tbl that were 
      %   removed.
      
      assert(istable(tbl));
      assert(isa(keymap,'containers.Map'));
      
      keys = tbl{:,keyfld};
      tfMovIdx = isa(keys,'MovieIndex');
      if tfMovIdx
        keys = int32(keys);
      end
      keysnew = arrayfun(@(x)keymap(x),keys);
      tfRm = keysnew==0;
      if tfMovIdx
        keysnew = MovieIndex(keysnew);
      end
      
      tblNew = tbl;
      tblNew.(keyfld) = keysnew;
      tblNew(tfRm,:) = [];
    end
    
    function tbl = emptyFLDSID()
      x = zeros(0,1);
      tbl = table(MovieIndex(x),x,x,'VariableNames',MFTable.FLDSID);
    end
        
    function tbl = emptyTable(varNames)
      % Create an empty MFTable 

      assert(strcmp(varNames{1},'mov'));
      n = numel(varNames);
      tbl = cell2table(cell(0,n),'VariableNames',varNames);
      tbl.mov = MovieIndex(tbl.mov);
    end
    
  end
  
  properties (Constant)
    % Uniquely IDs a frame/target
    FLDSID = {'mov' 'frm' 'iTgt'};
    
    % Core training/test data.
    FLDSCORE = {'mov' 'frm' 'iTgt' 'tfocc' 'p'};
    % In ROI case, .p is relative to .ROI
    FLDSCOREROI = {'mov' 'frm' 'iTgt' 'tfocc' 'p' 'roi'};

    FLDSFULL = {'mov' 'frm' 'iTgt' 'p' 'pTS' 'tfocc'};
    
    %  mov    single string unique ID for movieSetID combo
    %  frm    1-based frame index
    %  iTgt   1-based trx index
    %  p      Absolute label positions (px). Raster order: physpt,view,{x,y}
    %  pTS    [npts=nphyspt*nview] timestamps
    %  tfocc  [npts=nphyspt*nview] logical occluded flag
    %  pTrx   [nview*2], trx .x and .y. Raster order: view,{x,y}
    FLDSFULLTRX = {'mov' 'frm' 'iTgt' 'p' 'pTS' 'tfocc' 'pTrx'};    

    FLDSSUSP = {'mov' 'frm' 'iTgt' 'susp'};
  end
  
  methods (Static)
    
    function tbl = emptySusp()
      x = nan(0,1);
      tbl = table(x,x,x,x,'VariableNames',MFTable.FLDSSUSP);
    end
    
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
    
    function movIDs = formMultiMovieIDArray(movs)
      % movs: [nxnview] cellstr
      %
      % movIDs: [nx1] cellstr
      assert(iscellstr(movs) && ismatrix(movs));
      movIDs = arrayfun(@(x)MFTable.formMultiMovieID(movs(x,:)),...
        (1:size(movs,1))','uni',0);
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
    
    function tblMF = replaceMovieStrWithMovieIdx(tblMF,movIDs,movFull,tblDesc)
      % Replace localized movies or movieIDs with movie idxs
      %
      % movIDs: [nx1] cellstr movie IDs
      % movFull: [nx1] cellstr localized movies
      % tblDesc: string description of table (for warning messages)
      %
      % tblMF: .mov field updated to indices into movIDs/movFull. Warnings
      % thrown for unrecognized elements of .mov; these indices will be 0.
      
      szassert(movIDs,size(movFull));
      nrow = height(tblMF);
      assert(iscellstr(tblMF.mov) && size(tblMF.mov,2)==1);
      tblMFiMov = zeros(nrow,1); % indices into movIDs/movFull for each el of mov
      mapUnrecognizedMovies = containers.Map();
      for irow=1:nrow
        mov = tblMF.mov{irow};
        
        iMov = find(strcmp(mov,movIDs));
        if isscalar(iMov)
          tblMFiMov(irow) = iMov;
          continue;
        else
          assert(isempty(iMov),'Duplicate movieIDs found.');
        end
        
        iMov = find(strcmp(mov,movFull));
        if isscalar(iMov)
          tblMFiMov(irow) = iMov;
          continue;
        else
          assert(isempty(iMov),'Duplicate movie found.');
        end
        
        % If made it to here, mov is not recognized.
        assert(tblMFiMov(irow)==0);
        if ~mapUnrecognizedMovies.isKey(mov)
          warningNoTrace('MFTable:mov',...
            'Unrecognized movie in table ''%s'': %s',tblDesc,mov);
          mapUnrecognizedMovies(mov) = 1;
        end
      end
      
      tblMF.mov = tblMFiMov;
    end
          
  end
  
end  
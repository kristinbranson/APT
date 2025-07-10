classdef MFTable
% Movie/Frame/(Target) tables
%
% An MFTable is a table with cols 'mov' and 'frm' indicating movies and
% frame numbers.
  
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
    %  thetaTrx [nview], trx .theta
    FLDSFULLTRX = {'mov' 'frm' 'iTgt' 'p' 'pTS' 'tfocc' 'pTrx' 'thetaTrx'};    

    FLDSSUSP = {'mov' 'frm' 'iTgt' 'susp'};
  end
  
  methods (Static) % MFTables
    
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
      if isempty(tbl),
        tblNew = tbl;
        tfRm = false(0,1);
        return;
      end
      
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
      %tblNew(tfRm,:) = []; AL 20171002 ML subsasgn bug with movieIndex col
      tblNew = tblNew(~tfRm,:);
    end

    function tbl = emptyTable(varNames)
      % Create an empty MFTable with MovieIndex .movs

      assert(strcmp(varNames{1},'mov'));
      n = numel(varNames);
      x = nan(0,1);
      args = repmat({x},1,n);
      tbl = table(args{:},'VariableNames',varNames);
      tbl.mov = MovieIndex(tbl.mov);
    end

    function tbl = allocateTable(varNames,nrows)
      assert(strcmp(varNames{1},'mov'));
      n = numel(varNames);
      tbl = table('size',[nrows,n],'VariableNames',varNames,'VariableTypes',repmat(["double"],[1,n]));
      tbl.mov = MovieIndex(tbl.mov);
    end
    
    function tbl = emptySusp()
      x = nan(0,1);
      tbl = table(x,x,x,x,'VariableNames',MFTable.FLDSSUSP);
    end
    
    function tbl = intersectID(tbl,tblRestrict)
      % Restrict the rows of tbl to those also in tblRestrict, with respect
      % to MFTable.FLDSID.
      %
      % tbl, tblRestrict: tables with FLDSID
      %
      % tbl (out): that subset of tbl that is also in tblRestrict, ordered
      % as in tblRestrict.
      
      [~,~,ib] = intersect(...
        tblRestrict(:,MFTable.FLDSID),tbl(:,MFTable.FLDSID),'stable');
      tbl = tbl(ib,:);
    end
    
    function tf = isAfter(tbl,row)
      % follows canonical ordering
      % sortvars = {'mov' 'iTgt' 'frm'};
      % if isgt mismatch, false
      tf = sign(tbl.mov) == sign(row.mov);
      % if tbl.mov < row.mov, false
      tf = tf & abs(tbl.mov) >= abs(row.mov);
      idx = tbl.mov == row.mov;
      % for frames in the same movie, if tbl.frm < row.frm, false
      tf(idx) = tf(idx) & (tbl(idx,:).frm >= row.frm);
      % for labels on the same movie and frame, if tbl.iTgt < row.iTgt, false
      idx = idx & (tbl.frm == row.frm);
      if ~MFTable.isTgtUnset(row),
        tf(idx) = tf(idx) & (tbl(idx,:).iTgt >= row.iTgt);
        idx = idx & (tbl.iTgt == row.iTgt);
      end
      % exact same mft,false
      tf(idx) = false;

    end

    function tbl = sortCanonical(tbl)
      %assert(isa(tbl.mov,'MovieIndex'));
      tfgt = tbl.mov<0;
      tblGT = tbl(tfgt,:);
      tblReg = tbl(~tfgt,:);
      sortvars = {'mov' 'iTgt' 'frm'};
      tblReg = sortrows(tblReg,sortvars,{'ascend' 'ascend' 'ascend'});
      tblGT = sortrows(tblGT,sortvars,{'descend' 'ascend' 'ascend'});
      tbl = [tblReg; tblGT];
    end
    
    function [tblnew,tfNew] = tblDiff(tbl0,tbl1,varargin)
      % Remove entries of tbl1 in tbl0
      %
      % tbl0, tbl1: MF tables
      %
      % tblnew: new frames (rows of tbl0 whose (mov,frm,iTgt) are not in 
      %   tbl1)
      
      flds = myparse(varargin,...
        'flds',MFTable.FLDSID ...
        );

      tblnew = tbl0;
      tfNew = true(size(tbl0,1),1);
      if isempty(tbl1) || isempty(tbl0),
        return;
      end

      if ismember('iTgt',flds),
        iswild1 = MFTable.isTgtUnset(tbl1);
        fldsrest = setdiff(flds,'iTgt');

        if any(iswild1),
          % if tbl0 has movie m, frame f, target *
          % and tbl1 has movie m, frame f, target 1
          % tfNew = true
          % if tbl0 has movie m, frame f, target *
          % and tbl1 has movie m, frame f, target *
          % tfNew = false

          tfNew = tfNew & ~tblismember(tbl0,tbl1(iswild1,:),fldsrest);
        end
      else
        iswild1 = false(size(tbl1,1),1);
      end

      tfNew = tfNew & ~tblismember(tbl0,tbl1(~iswild1,:),flds);
      tblnew = tbl0(tfNew,:);
    end
    
    function [tblPnew,tblPupdate,idx0update] = tblPDiff(tblP0,tblP)
      % Compare tblP to tblP0 wrt fields FLDSCORE (see below)
      %
      % tblP0, tblP: MF tables
      %
      % tblPNew: new frames (rows of tblP whose (mov,frm,iTgt) are not in 
      %   tblP0)
      % tblPupdate: existing frames with new positions/tags (rows of tblP
      %   whose (mov,frm,iTgt) are in tblP0, but whose p or tfocc differ
      % idx0update: indices into rows of tblP0 corresponding to tblPupdate;
      %   ie tblP0(idx0update,:) ~ tblPupdate
      
      unsetval = -1;

      FLDSID = MFTable.FLDSID;
      FLDSCORE = MFTable.FLDSCORE;
      tblfldscontainsassert(tblP0,FLDSCORE);
      tblfldscontainsassert(tblP,FLDSCORE);
      
      copytblP0 = tblP0;
      copytblP0.iTgt(MFTable.isTgtUnset(tblP0)) = unsetval;
      copytblP = tblP;
      copytblP.iTgt(MFTable.isTgtUnset(tblP)) = unsetval;
      
      tfNew = ~tblismember(copytblP,copytblP0,FLDSID);
      [tfSame,locSame] = tblismember(tblP,tblP0,FLDSCORE);
      if nnz(tfSame)>0
        % side check, all shared fields must be identical for 'same' rows
        fldsshared = intersect(tblflds(tblP),tblflds(tblP0));
        assert(isequaln(tblP(tfSame,fldsshared),...
                        tblP0(locSame(tfSame),fldsshared)));
      end
      tfDiff = ~tfSame;
      tfUpdate = tfDiff & ~tfNew; 

      tblPnew = tblP(tfNew,:);
      tblPupdate = tblP(tfUpdate,:);            
     
      [tf,loc] = tblismember(tblPupdate,tblP0,FLDSID);
      assert(all(tf));
      idx0update = loc;
    end

    function movID = formMultiMovieID(movs)
      % Form multimovie char ID
      %
      % WARNING: This only warns if one of movs contains the ID separator.
      % Use at your own risk.
      % 
      % movs: row cellstr vec
      %
      % movID: char
      
      assert(iscellstr(movs) && isrow(movs));
      if exist('contains','builtin')>0 && any(contains(movs,'#')) || ...
         any(~cellfun(@isempty,regexp(movs,'#','once')))
        warningNoTrace('Movies contain ID separator ''#''.');
      end
        
      movID = sprintf('%s#',movs{:});
      movID = movID(1:end-1);
    end
    
    function movIDs = formMultiMovieIDArray(movs,varargin)
      % movs: [nxnview] cellstr
      %
      % movIDs: [nx1] cellstr
      
      [separator,checkseparator] = myparse(varargin,...
        'separator','#',...
        'checkseparator',true);
      
      assert(iscellstr(movs) && ismatrix(movs));
      
      if checkseparator
        tferr = exist('contains','builtin')>0 && any(contains(movs(:),separator)) ...
                || any(~cellfun(@isempty,strfind(movs(:),separator)));  
        % contains introed in 16b
        if tferr
          error('Movies contain ID separator ''%s''.',separator);
        end
      end
      
      nvw = size(movs,2);
      movIDs = movs(:,1);
      for ivw=2:nvw
        movIDs = strcat(movIDs,separator,movs(:,ivw));
      end
    end
    
    function movs = unpackMultiMovieID(movID)
      movs = regexp(movID,'#','split');
    end
    
%     function I = fetchImages(tMF)
%       %
%       % tMF: [nxncol] MFTable. tMF.mov: [nxnview] cellstr
%       %
%       % I: [nxnview]
%       %
%       % PROB REMOVE ME, dup of CPRData.getFrames. No callsites in APT
%       % application. Unsafe wrt preloading.
%       
%       movIDs = MFTable.formMultiMovieIDArray(tMF.mov); % errs if any ID separator issues
%       [movIDsUn,idx] = unique(movIDs);
%       
%       % open moviereaders
%       movsetsUn = tMF.mov(idx,:);
%       [nMovsetsUn,nView] = size(movsetsUn);
%       mrcell = cell(size(movsetsUn));
%       for iMovSet=1:nMovsetsUn
%         for iView=1:nView
%           mr = MovieReader();
%           mr.open(movsetsUn{iMovSet,iView});
%           mr.forceGrayscale = true;
%           mrcell{iMovSet,iView} = mr;
%         end
%         fprintf(1,'MovieSet %d moviereader.\n',iMovSet);
%       end
%       
%       fprintf('%d unique moviesets.\n',nMovsetsUn);
%       
%       nRows = size(tMF,1);
%       I = cell(nRows,nView);
%       for iRow=1:nRows
%         frm = tMF.frm(iRow);
%         id = movIDs{iRow};
%         iMovSet = strcmp(id,movIDsUn);
%         assert(nnz(iMovSet)==1);
%         for iView=1:nView
%           I{iRow,iView} = mrcell{iMovSet,iView}.readframe(frm);
%         end
%         if mod(iRow,10)==0
%           fprintf(1,'Read images: row %d\n',iRow);
%         end
%       end
%     end
%     
%     function I = fetchImagesSafeVideoRdr(tMF,varargin)
%       % No callsites in APT app
%       
%       [movFld] = myparse(varargin,...
%         'movFld','mov');
%       
%       n = height(tMF);
%       movs = tMF.(movFld);
%       nview = size(movs,2);      
%       I = cell(size(movs));
%       
%       movsUn = unique(movs(:));
%       nMovsUn = numel(movsUn);
%       
%       fprintf('n=%d, nview=%d, nMovsUn=%d\n',n,nview,nMovsUn);
%       
%       for iMov=1:nMovsUn
%         m = movsUn{iMov};
%         
%         tf = strcmp(movs,m);
%         [rows,vws] = find(tf);
%         nIm = numel(rows);
%         maxfrm = max(tMF.frm(rows));
%         fprintf(1,'movUn %d (%s). %d images to read. maxfrm is %d.\n',...
%           iMov,m,nIm,maxfrm);
%         
%         imstack = readAllFrames(m,maxfrm);
%         
%         for iIm=1:nIm
%           trow = tMF(rows(iIm),:);
%           %im = mr.readframe(trow.frm);
%           im = imstack{trow.frm};
%           I{rows(iIm),vws(iIm)} = im;
%         end
%       end
%     end
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
    
    function tblMF = loadTableFromMatfile(fname)
      s = load(fname,'-mat');
      flds = fieldnames(s);
      tf = structfun(@(x)istable(x) && all(tblfldscontains(x,MFTable.FLDSID)),s);
      fldsTbl = flds(tf);
      switch numel(fldsTbl)
        case 0
          error('No movie-frame-target tables found in file ''%s''.',fname);
        case 1
          tblMF = s.(fldsTbl{1});
        otherwise
          error('Multiple movie-frame-target tables found in file ''%s''.',fname);
      end
    end

    function v = isTgtUnset(tblMFT)
      v = isnan(tblMFT.iTgt) | (tblMFT.iTgt==0);
    end

    function tblMFT = unsetTgt(tblMFT)
      tblMFT.iTgt(:) = nan;
      tblMF = removevars(tblMFT,'iTgt');
      [~,idx] = unique(tblMF);
      tblMFT = tblMFT(idx,:);
    end
          
  end
  
end  
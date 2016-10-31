classdef MFTable
  % MFTable -- Movie/Frame tables
  %
  % An MFTable is a table with cols 'mov' and 'frm' indicating movies and
  % frame numbers. 
  
  methods (Static)
    function [tblPnew,tblPupdate,idx0update] = tblPDiff(tblP0,tblP)
      % Compare tblP to tblP0
      %
      % tblP0, tblP: MF tables
      %
      % tblPNew: new frames (rows of tblP whose movie-frame ID are not in tblP0)
      % tblPupdate: existing frames with new positions/tags (rows of tblP
      %   whos movie+frame ID are in tblP0, but whose eg p field is different).
      % idx0update: indices into rows of tblP0 corresponding to tblPupdate;
      %   ie tblP0(idx0update,:) ~ tblPupdate
      
      tblMF0 = tblP0(:,{'mov' 'frm'});
      tblMF = tblP(:,{'mov' 'frm'});
      tfPotentiallyUpdatedRows = ismember(tblMF,tblMF0);
      tfNewRows = ~tfPotentiallyUpdatedRows;
      
      tblPnew = tblP(tfNewRows,:);
      tblPupdate = tblP(tfPotentiallyUpdatedRows,:);
      tblPupdate = setdiff(tblPupdate,tblP0);
      
      [tf,loc] = ismember(tblPupdate(:,{'mov' 'frm'}),tblMF0);
      assert(all(tf));
      idx0update = loc(tf);
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
      
      tMF = tblMF(:,{'mov' 'frm'});
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
  
  
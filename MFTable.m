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
  end
  
end
  
  
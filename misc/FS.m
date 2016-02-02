classdef FS
  
  properties (Constant)
    % TRPAT = '(?<td>.*)@(?<tdnote>.*)@
  end
  methods (Static)
    
    function [type,info] = parsename(n)
      toks = regexp(n,'@','split');
      ntok = numel(toks);
      switch ntok
        case 3
          type = 'tp';
          info = cell2struct(toks(2:end),{'name' 'date'},2);
        case 4
          type = toks{1};
          switch type 
            case 'td'
              info = cell2struct(toks(2:end),{'name' 'note' 'date'},2);
            case 'tdI'
              info = cell2struct(toks(2:end),{'tdname' 'note' 'date'},2);
          end
        case 7
          type = 'tr';
          info = cell2struct(toks,{'tdname' 'tdnote' 'tdIname' 'tdIvar' 'tpname' 'sha' 'date'},2);
        otherwise
          assert(false);
      end
      
    end
    
    function n = formTrainedClassifierName(tdname,tdIname,tdIvar,tpname,sha)
      [~,s_td] = FS.parsename(tdname);
      if isempty(tdIname)
        tdI_note = '';
      else
        [~,s_tdI] = FS.parsename(tdIname);
        tdI_note = s_tdI.note;
      end
      [~,s_tp] = FS.parsename(tpname);
      
      n = sprintf('%s@%s@%s@%s@%s@%s@%s',s_td.name,s_td.note,tdI_note,tdIvar,...
        s_tp.name,sha,datestr(now,'mmddTHHMM'));
    end
    
    function n = formTestResultsFolderName(trname,tdname,tdIname,tdIvar)
      [~,s_tr] = FS.parsename(trname);
      [~,s_td] = FS.parsename(tdname);
      if isempty(tdIname)
        tdI_note = '';
      else
        [~,s_tdI] = FS.parsename(tdIname);
        tdI_note = s_tdI.note;
      end
      
      n = sprintf('%s@%s@%s@%s@%s__%s@%s@%s@%s__%s',...
        s_tr.tdname,s_tr.tdnote,s_tr.tdIname,s_tr.tdIvar,s_tr.tpname,...
        s_td.name,s_td.note,tdI_note,tdIvar,...
        datestr(now,'mmddTHHMM'));      
    end
    
  end
  
end
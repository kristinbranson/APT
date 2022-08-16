classdef String
  % String utils  

  % comma-separated list of items
  methods (Static)
    
    function c = commaSepList2CellStr(s)
      c = regexp(s,',','split');
      c = strtrim(c);
    end
    
    function s = cellstr2CommaSepList(c)
      if isempty(c)
        s = '';
      else
        s = [sprintf('%s,',c{1:end-1}) c{end}];
      end
    end
    
    function s = cellstr2DelimList(c,d)
      if isempty(c)
        s = '';
      else
        pat = sprintf('%%s%s',d);
        s = [sprintf(pat,c{1:end-1}) c{end}];
      end
    end

    function s = quoteCellStr(c,q)
      if nargin < 2,
        q = '"';
      end
      s = cellfun(@(x) [q x q],c,'Uni',0);
    end
    
    function s = escapeSpaces(s)
      s = regexprep(s,'([^\\]) ','$1\\ ');
    end

    function outcmd = escapeQuotes(incmd)
      outcmd = strrep(strrep(incmd,'\','\\'),'"','\"');
    end
    
    % see civilizedStringFromCellArrayOfStrings
    
    function s = niceUpperCase(s)
      s = lower(s);
      if ~isempty(s)
        s(1) = upper(s(1));
      end
    end
      
      
  end
  
end
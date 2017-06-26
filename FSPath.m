classdef FSPath
  % FSPath -- File System Path utilities
  
  % STANDARDIZED paths use the separator '/' and eliminate
  % double-separators etc. Standardized paths are supposed to look the same 
  % across platforms, modulo differing filesystem mount names etc.
  %
  % MACROS begin look like $datadir and be substituted.   
  
  methods (Static)
  
    function [stdfull,stdmed,stdshort] = standardPath(p)
      % Like standardPathChar
      %
      % p: char or cellstr
      %
      % stdfull, stdmed, stdshort: char/cellstr per p
      
      if ischar(p)
        [stdfull,stdmed,stdshort] = FSPath.standardPathChar(p);
      else
        [stdfull,stdmed,stdshort] = cellfun(@FSPath.standardPathChar,p,'uni',0);
      end
    end
    
    function [stdfull,stdmed,stdshort] = standardPathChar(p)
      % Take a fullpath and construct standardized/platform-indep paths.
      % Standard paths use / as a separator.
      %
      % p: char
      %
      % stdfull, stdmed: stdshort: standardized full, med, short paths
      
      p = regexprep(p,'\\','/');

      %tfLeadingDoubleSlash = numel(p)>=2 && strcmp(p(1:2),'//');      
      pnew = regexprep(p,'//','/');
      while ~strcmp(pnew,p)
        p = pnew;
        pnew = regexprep(p,'//','/');
      end
%       if tfLeadingDoubleSlash
%         p = ['/' p];
%       end
      
      stdfull = p;
      [upper,stdshort] = myfileparts(p);
      [~,upperlast] = myfileparts(upper);
      stdmed = [upperlast '/' stdshort];      
    end
    
    function str = macroReplace(str,sMacro)
      % sMacro: macro struct
      
      macros = fieldnames(sMacro);
      for i=1:numel(macros)
        mpat = ['\$' macros{i}];
        val = sMacro.(macros{i});
        val = regexprep(val,'\\','\\\\');
        str = regexprep(str,mpat,val);
      end
    end
    
    function str = fullyLocalizeStandardizeChar(str,sMacro)
      str = FSPath.macroReplace(str,sMacro);
      str = FSPath.standardPathChar(str);
      str = FSPath.platformizePath(str);
    end
    
    function s = fullyLocalizeStandardize(s,sMacro)
      % s: cellstr
      s = cellfun(@(x)FSPath.fullyLocalizeStandardizeChar(x,sMacro),s,'uni',0);
    end
      
    function str = platformizePath(str,ispcval)
      % Depending on platform, replace / with \ or vice versa
      
      assert(ischar(str));
      
      if exist('ispcval','var')==0
        ispcval = ispc();
      end
      
      if ispcval
        str = regexprep(str,'/','\');
        if numel(str)>=2 && strcmp(str(1),'\') && ~strcmp(str(2),'\')
          str = ['\' str];
        end
      else
        str = regexprep(str,'\\','/');
      end
    end
    
    function tf = hasMacro(str)
      tf = ~isempty(regexp(str,'\$','once'));
    end
    
    function warnUnreplacedMacros(strs)
      if ~isempty(strs)
        toks = cellfun(@(x)regexp(x,'\$([a-zA-Z]+)','tokens'),strs,'uni',0);
        toks = [toks{:}];
        toks = [toks{:}];
        if ~isempty(toks)
          toks = unique(toks);
          cellfun(@(x)warningNoTrace('FSPath:macro','Unreplaced macro: $%s',x),toks);
        end
      end
    end
    
    function errUnreplacedMacros(strs)
      strs = cellstr(strs);
      toks = cellfun(@(x)regexp(x,'\$([a-zA-Z0-9_]+)','tokens'),strs,'uni',0);
      toks = [toks{:}];
      toks = [toks{:}];
      if ~isempty(toks)
        tokstr = String.cellstr2CommaSepList(toks);
        error('FSPath:macro','Unreplaced macros: $%s',tokstr);
      end
    end
    
    function parts = fullfileparts(p)
      % Break up a path into its parts
      % 
      % p: a path
      %
      % parts: row cellstr, fully broken-up path
      
      % Some bizzare edge cases can occur with eg p=='///a/b//'
      
      parts = cell(1,0);
      while ~isempty(p) && ~strcmp(p,'\') 
        [pnew,fe] = myfileparts(p);
        if isempty(fe) && strcmp(pnew,p)          
          parts{1,end+1} = p; %#ok<AGROW>
          p = '';
        elseif isempty(fe) && ~strcmp(pnew,p)
          p = pnew;
        else % ~isempty(fe)
          parts{1,end+1} = fe; %#ok<AGROW>
          p = pnew;
        end        
      end
      parts = parts(end:-1:1);
    end
    
    function base = commonbase(p)
      % Find common base of paths
      %
      % p: cellstr of paths
      %
      % base: char, common base path. Will be '' if there is no base path.
      
      nchar = cellfun(@numel,p);
      n = min(nchar);
      base = '';
      for i=1:n
        c = cellfun(@(x)x(i),p);
        if all(c==c(1))
          base(1,end+1) = c(1); %#ok<AGROW>
        else
          break;
        end
      end
    end
      
  end
  
end
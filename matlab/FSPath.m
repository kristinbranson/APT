classdef FSPath
% Filesystem path utilities
%  
% STANDARDIZED paths use the separator '/' and eliminate
% double-separators etc. Standardized paths are supposed to look the same 
% across platforms, modulo differing filesystem mount names etc.
%
% MACROS look like $datadir and can be replaced/substituted.
  
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
    
    function result = macroReplace(input, sMacro)
      % Replaces any macros present in input with the values given in sMacro.
      % input: string or cell array of strings
      % sMacro: macro struct
      
      if iscell(input)
        result = cellfun(@(s)(FSPath.macroReplace(s,sMacro)), ...
                         input, ...
                         'UniformOutput', false) ;
      else
        % if input is a single string
        result = input ;
        macros = fieldnames(sMacro);
        for i=1:numel(macros)
          macro = macros{i} ;
          mpat = ['\$' macro];
          val = sMacro.(macro);
          val = regexprep(val,'\\','\\\\');
          result = regexprep(result,mpat,val);
        end
      end
    end  % function
    
    function result = fullyLocalizeStandardizeChar(str0,sMacro)
      str1 = FSPath.macroReplace(str0,sMacro);
      str2 = FSPath.standardPathChar(str1);
      result = FSPath.platformizePath(str2);
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
    
    function tf = hasAnyMacro(str)
      tf = ~isempty(regexp(str,'\$','once'));
    end
    function tf = hasMacro(str,macro)
      tf = ~isempty(regexp(str,['\$' macro],'once'));
    end
    
    function [macros,pathstrMacroized] = macrosPresent(pathstr,sMacro)
      % Check to see if any macros in sMacro could apply to pathstr, ie if 
      % any macro values are present in pathstr. If so, return which macros 
      % apply, and return the macroized pathstr.
      %
      % pathstr: char
      % sMacro: macro struct
      % 
      % macros: [nmatchx1] cellstr. Macros that matched. nmatch==0 if there
      % are no matches.
      % pathstrmacroized: [nmatchx1] cellstr. Macroized versions of
      % pathstr, where matching strings have been replaced with $<macro>. 
      %
      % Example: pathstr='/path/to/data/exp1/mov.avi', 
      %   sMacro.dataroot=/path/to/data. Then
      %
      %   - macros = {'dataroot'}
      %   - pathstrmacroized = {'$dataroot/exp1/mov.avi'}
      
      macrosAll = fieldnames(sMacro);
      macros = cell(0,1);
      pathstrMacroized = cell(0,1);
      for m=macrosAll(:)',m=m{1}; %#ok<FXSET>
        val = sMacro.(m);
        pat = regexprep(val,'\\','\\\\');
        if ispc
          idx = regexpi(pathstr,pat,'once');
        else
          idx = regexp(pathstr,pat,'once');
        end
        if ~isempty(idx)
          macros{end+1,1} = m; %#ok<AGROW>
          if ispc
            pathstrMacroized{end+1,1} = regexprep(pathstr,pat,['$' m],'ignorecase'); %#ok<AGROW>
          else
            pathstrMacroized{end+1,1} = regexprep(pathstr,pat,['$' m]); %#ok<AGROW>
          end
        end
      end
    end 
    
    function [tfCancel,macro,pathstrsMacroized] = ...
                                    offerMacroization(sMacro,pathstrs)
      % If any macros are present in all pathstrs, let user optionally
      % select macroized versions of pathstrs. Macroization can only be
      % done with a single macro that is common to all pathstrs.
      %
      % sMacro: macro struct
      % pathstrs: cellstr (can be nonscalar)
      %
      % tfCancel: if true, user canceled
      % macro: macro used/replaced, or [] if no macro found/selected.
      % pathstrsMacroized: same size as pathstrs; macroized pathstrs (using
      %   macro), or original pathstrs if macro is [].

      assert(iscellstr(pathstrs) && isvector(pathstrs));
      
      [macroMatchCell,pathstrMacroizedCell] = ...
        cellfun(@(zzP)FSPath.macrosPresent(zzP,sMacro),pathstrs,'uni',0);
      macrosMatch = macroMatchCell{1};
      npstrs = numel(pathstrs);
      for i=2:npstrs
        macrosMatch = intersect(macrosMatch,macroMatchCell{i});
      end
      if isempty(macrosMatch)
        tfCancel = false;
        macro = [];
        pathstrsMacroized = pathstrs;
      else
        % 1+ macros are common to all pathstrs
        [tf,loc] = ismember(macrosMatch,macroMatchCell{1});
        assert(all(tf));
        pathstr1Macroized = pathstrMacroizedCell{1}(loc);
        liststr = [{pathstrs{1}};pathstr1Macroized];
        [sel,ok] = listdlg('ListString',liststr,...
          'SelectionMode','single',...
          'ListSize',[700 200],...
          'Name','Macros Available',...
          'PromptString','Select optional macro to use for moviefile(s):');
        tfCancel = ~ok;
        if ok
          if sel==1
            macro = [];
            pathstrsMacroized = pathstrs;
          else
            macro = macrosMatch{sel-1};
            pathstrsMacroized = cell(size(pathstrs));
            for i=1:npstrs
              [tf,loc] = ismember(macro,macroMatchCell{i});
              assert(tf);
              pathstrsMacroized{i} = pathstrMacroizedCell{i}{loc};
            end
          end
        else
          macro = [];
          pathstrsMacroized = [];
        end
      end
    end
    
    function [tfMatch,trxFileMacroized] = ...
                      tryTrxfileMacroization(trxFile,movdir)
      % Try to match standard pattern <movdir>/<trxFile>
      %
      % trxFile: full path to trxfile (no macros)
      % movdir: full path to movie directory
      %
      % tfMatch: if true, trxFile looks like <movdir>/trxFileShort (with 
      %   appropriate filesep)
      % trxFileMacroized: If tfMatch is true, then $movdir/trxFileShort 
      %   (with appropriate filesep). Indeterminate otherwise
      
      tfile = FSPath.standardPathChar(trxFile);
      movdir = FSPath.standardPathChar(movdir);
      nmovdir = numel(movdir);
      if numel(tfile)>nmovdir
        if ispc
          tfMatch = strncmpi(tfile,movdir,nmovdir);
        else
          tfMatch = strncmp(tfile,movdir,nmovdir);
        end
        tfMatch = tfMatch && tfile(nmovdir+1)=='/';
      else
        tfMatch = false;
      end
      
      if tfMatch
        tfileS = tfile(nmovdir+2:end);
        %[~,tfileS] = myfileparts(tfile);
        trxFileMacroized = FSPath.platformizePath(['$movdir/' tfileS]);
      else
        trxFileMacroized = [];
      end
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
    
    function estr = errStrFileNotFound(file,fileType)
      estr = sprintf('Cannot find %s ''%s''.',fileType,file);
    end
    function estr = errStrFileNotFoundMacroAware(file,fileFull,fileType)
      % Macro-aware file-not-found error string. 
      % Backslash (\) is NOT ESCAPED.
      %
      % file: raw file possibly with macros
      % fileFull: macro-replaced file
      % fileType: eg 'movie' or 'trxfile'
      
      if ~FSPath.hasAnyMacro(file)
        estr = sprintf('Cannot find %s ''%s''.',fileType,file);
      else
        estr = sprintf('Cannot find %s ''%s'', macro-expanded to ''%s''.',...
          fileType,file,fileFull);
      end
    end
    function errDlgFileNotFound(estr)
      errordlg(estr,'File not found');
    end
    function throwErrFileNotFoundMacroAware(file,fileFull,fileType)
      emsg = FSPath.errStrFileNotFoundMacroAware(file,fileFull,fileType);
      errorNoTrace(emsg);
    end
    
    function str = fileStrMacroAware(file,fileFull)
      if ~FSPath.hasAnyMacro(file)
        str = file;
      else
        str = sprintf('%s, macro-expanded to %s.',file,fileFull);
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
    
    function fname2 = twoLevelFilename(fname)
      % eg folder/file.txt
      [path0,base] = myfileparts(fname);
      [~,parent] = fileparts(path0);
      fname2 = fullfile(parent,base);
    end
        
    function fname = nLevelFname(fname,n)
      % fname = '/path/to/a/file.ext'
      % 
      % nLevelFname(fname,1) is 'file.ext'
      % nLevelFname(fname,2) is 'a/file.ext'
      % ...
      fname = cellfun(@(x)FSPath.nLevelFnameChar(x,n),fname,'uni',0);
    end
    function fnameN = nLevelFnameChar(fname,n)
      fnameN = '';
      while n>0
        [fname,base] = myfileparts(fname);
        if isempty(base)
          fnameN = fullfile(fname,fnameN);
          fname = '';
        else
          fnameN = fullfile(base,fnameN);
        end
        n = n-1;
      end
    end

    function pbase = maxExistingBasePath(p)
      % Find maximum existing basepath of p
      %
      % p: full path
      %
      % pbase: maximmal base path of p that exists. Can be ''.
      
      pbase = p;
      while ~isempty(pbase) && exist(pbase,'dir')==0
        pbase2 = fileparts(pbase);
        if strcmp(pbase2,pbase)
          pbase2 = '';
        end
        pbase = pbase2;
      end
    end
    
    function base = commonbase(p,mindepth)
      % Find common base of paths
      %
      % p: cellstr of paths
      %
      % base: char, common base path. Will be empty if there is no base path.
      if nargin < 2,
        mindepth = 0;
      end
%       assert(isunix);
      assert(~isempty(p) && iscellstr(p));
%       if ~all(cellfun(@(x)x(1)=='/',p))
%         error('All paths must be absolute unix paths.');
%       end
      
      for i = 1:numel(p),
        if p{i}(end) == filesep,
          p{i} = p{i}(1:end-1);
        end
      end

      if isscalar(p)
        if exist(p{1},'dir'),
          base = p{1};
        else
          assert(exist(p{1},'file')>0);
          base = fileparts(p{1});
        end
        %base = p{1};
        return;
      end
      
      parts = cellfun(@FSPath.fullfileparts,p,'uni',0);
      % parts{i} is a row vec of folders
      nfldrs = cellfun(@numel,parts);
      n = min(nfldrs);

      if ispc
        cmpFcn = @(x)all(strcmpi(x,x{1}));
      else
        cmpFcn = @(x)isequal(x{:});
      end
      
      for i=1:n
        c = cellfun(@(x)x{i},parts,'uni',0);
        if cmpFcn(c)
          % none; all ith folders match
        else
          i = i-1; %#ok<FXSET>
          break;
        end
      end
      % parts/folders match up to and including position i
      
      if i==0
        base = '';
      elseif ispc
        base = fullfile(parts{1}{1:i});
      else
        base = ['/' fullfile(parts{1}{1:i})];
      end
      
      if mindepth > 0,
        if i < mindepth,
          base = unique( cellfun(@(x) fullfile(x{1:min(numel(x),mindepth)}), parts,'Uni',0) );
          if ~ispc,
            base = cellfun(@(x) ['/',x],base,'Uni',0);
          end
        else
          base = {base};
        end
      end
    end  % function

    function result = replacePrefix(path, oldPrefix, newPrefix)
      % Replace the prefix in a single path.
      % Just does string replacement, so should work with any style of path.
      if startsWith(path, oldPrefix) ,
        oldPrefixLength = strlength(oldPrefix) ;
        stem = extractAfter(path, oldPrefixLength) ;
        result = strcat(newPrefix, stem) ;
      else
        result = path ;
      end
    end  % function 
      
    function result = replaceExtension(path, newExtension)
      % Replace the extension of path with newExtension.  newExtension should
      % include the '.'.  This uses fileparts(), so should only be used with paths
      % that are appropriate for the frontend platform.
      [parent, base] = fileparts(path) ;
      result = fullfile(parent, strcat(base, newExtension)) ;
    end  % function 
  end  % methods (Static) 
end  % classdef

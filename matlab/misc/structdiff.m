function [fdiff,fnot1,fnot2] = structdiff(s1,s2,varargin)
% [fdiff,fnot1,fnot2] = structdiff(s1,s2,varargin)
%
% fdiff: cellstr, fields present in s1 and s2 that differ
% fnot1: cellstr, fields not present in s1 but present in s2
% fnot2: etc
%
% Output args apply only to top-level fields.
% 
% Optional PVs: 
%   pfix. Default ''.
%   obj2struct. Automatically structify objects. Default true.
%   fcn2char. Convert function handles to strings (eg to avoid differences
%     in workspaces). Default false.
%   quiet. Default false.

assert(isscalar(s1));
assert(isscalar(s2));

[pfix,obj2struct,fcn2char,quiet,showdiff] = myparse(varargin,...
  'pfix','',...
  'obj2struct',true,...
  'fcn2char',false,...
  'quiet',false,...
  'showdiff',true...
  );

warnst = warning('off','MATLAB:structOnObject');
if isobject(s1) && obj2struct
  s1 = struct(s1);
end
if isobject(s2) && obj2struct
  s2 = struct(s2);
end
warning(warnst);

f1 = fieldnames(s1);
f2 = fieldnames(s2);
fnot2 = setdiff(f1,f2);
fnot1 = setdiff(f2,f1);
if ~isempty(fnot1) && ~quiet
  cellfun(@(x)fprintf(1,'%s.%s: not present in s1:\n',pfix,x),fnot1);
end
if ~isempty(fnot2) && ~quiet
  cellfun(@(x)fprintf(1,'%s.%s: not present in s2:\n',pfix,x),fnot2);
end

fboth = intersect(f1,f2);
fdiff = cell(0,1);
for f = fboth(:)',f=f{1}; %#ok<FXSET>
  v1 = s1.(f);
  v2 = s2.(f);
  
  if fcn2char && isa(v1,'function_handle') && isa(v2,'function_handle')
    v1 = char(v1);
    v2 = char(v2);
  end    
    
  if isequaln(v1,v2)
    % none
  elseif ~isequal(size(v1),size(v2))
    if ~quiet
      fprintf(1,'%s.%s: SIZE DIFFERS\n',pfix,f);
    end
    fdiff{end+1,1} = [pfix '.' f]; %#ok<AGROW>
  elseif isstruct(v1)&&isstruct(v2)&&numel(v1)==numel(v2) || ...
         isobject(v1)&&isobject(v2)&&numel(v1)==numel(v2)&&obj2struct       
    if isobject(v1)
      v1 = lclStructize(v1);
    end
    if isobject(v2)
      v2 = lclStructize(v2);
    end
    
    nv = numel(v1);    
    if nv==1
      newpfix = [pfix '.' f];
      fdiffsub = structdiff(v1,v2,'pfix',newpfix,'obj2struct',obj2struct,...
        'fcn2char',fcn2char,'quiet',quiet,'showdiff',showdiff);
      fdiff = [fdiff; fdiffsub]; %#ok<AGROW>
    else
      for i = 1:nv
        newpfix = sprintf('%s.%s(%d)',pfix,f,i);
        fdiffsub = structdiff(v1(i),v2(i),'pfix',newpfix,'obj2struct',obj2struct,...
          'fcn2char',fcn2char,'quiet',quiet,'showdiff',showdiff);
        fdiff = [fdiff; fdiffsub]; %#ok<AGROW>
      end
    end
  else
    if ~quiet
      if showdiff
        str1 = lclToChar(v1);
        str2 = lclToChar(v2);
        fprintf(1,'%s.%s: %s -> %s\n',pfix,f,str1,str2);
      else
        fprintf(1,'%s.%s: DIFFER\n',pfix,f);
      end
    end
    fdiff{end+1,1} = [pfix '.' f]; %#ok<AGROW>
  end
end

function s = lclToChar(v)
if ischar(v)
  s = v;
elseif isnumeric(v)
  s = mat2str(v);
else
  s = '<unk>';
end

function s = lclStructize(o)
assert(isobject(o));
s = cell(size(o));
warnst = warning('off','MATLAB:structOnObject');
for i = 1:numel(o)
  s{i} = struct(o(i));
end
warning(warnst);
s = cell2mat(s);
    
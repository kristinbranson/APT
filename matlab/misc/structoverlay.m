function [s,baseused] = structoverlay(sbase,sover,varargin)
% [s,baseused] = structoverlay(sbase,sover,varargin)
% Overlay 'leaf nodes' of sover onto sbase
%
% sbase: scalar base struct
% sover: scalar overlay struct
%
% s: scalar struct, result of overlay
% baseused: cellstr of 'paths' specifying fields where sbase values were
% used/retained
%
% optional PVs:
% - 'path'. String, defaults to ''. Current struct "path", eg
% .topfield.subfield.
% - 'dontWarnUnrecog'. Logical scalar, defaults to false. If true, don't 
% throw unrecognized field warning.
% - 'allowedUnrecogFlds'. Cellstr, fields that may not be in sbase but
% which will be included in s

[path,dontWarnUnrecog,allowedUnrecogFlds] = myparse(varargin,...
  'path','',...
  'dontWarnUnrecog',false,...
  'allowedUnrecogFlds',cell(1,0));
fldsBase = fieldnames(sbase);
fldsOver = fieldnames(sover);
baseused = setdiff(fldsBase,fldsOver);
baseused = strcat(path,'.',baseused(:));

for f = fldsOver(:)',f=f{1}; %#ok<FXSET>
  newpath = [path '.' f];
  if ~isfield(sbase,f)
    if isequal(allowedUnrecogFlds,'all') || any(strcmp(f,allowedUnrecogFlds))
      sbase.(f) = sover.(f);
    else
      if ~dontWarnUnrecog
        warning('structoverlay:unrecognizedfield','Ignoring unrecognized field ''%s''.',...
         newpath);
      end
    end
  elseif isstruct(sbase.(f)) && isstruct(sover.(f))
    assert(isscalar(sbase.(f)));
    % Allow nonscalar sover.(f)
    numOver = numel(sover.(f));
    assert(numOver>0);
    sbaseEl = sbase.(f);
    sbase.(f) = [];
    for i=1:numOver
      newpathindexed = [newpath sprintf('(%d)',i)];
      [newEl,newBU] = structoverlay( ...
        sbaseEl,sover.(f)(i),...
        'path',newpathindexed,...
        'dontWarnUnrecog',dontWarnUnrecog,...
        'allowedUnrecogFlds',allowedUnrecogFlds);
      sbase.(f) = structappend(sbase.(f),newEl,1);
      baseused = [baseused;newBU]; %#ok<AGROW>
    end
  elseif isstruct(sbase.(f)) && ~isstruct(sover.(f))
    warningNoTrace('structoverlay:nonstruct','Ignoring non-struct value of ''%s''.',...
      newpath );
    baseused{end+1,1} = newpath; %#ok<AGROW>
  else % sbase.(f) is not a struct
    sbase.(f) = sover.(f);
    % sover.(f) might be the same value as sbase.(f), but we don't consider
    % that base is being used in that case
    %
    % if sover.(f) is a struct that might be another special case where
    % sometimes we want to warn etc.
  end
end

s = sbase;
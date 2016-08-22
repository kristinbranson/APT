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

[path,dontWarnUnrecog] = myparse(varargin,...
  'path','',...
  'dontWarnUnrecog',false);
fldsBase = fieldnames(sbase);
fldsOver = fieldnames(sover);
baseused = setdiff(fldsBase,fldsOver);
baseused = strcat(path,'.',baseused(:));

for f = fldsOver(:)',f=f{1}; %#ok<FXSET>
  newpath = [path '.' f];
  if ~isfield(sbase,f) 
    if ~dontWarnUnrecog
      warning('structoverlay:unrecognizedfield','Ignoring unrecognized field ''%s''.',...
        newpath);
    end
  elseif isstruct(sbase.(f)) && isstruct(sover.(f))
    [sbase.(f),tmpBU] = structoverlay(sbase.(f),sover.(f),'path',newpath,...
      'dontWarnUnrecog',dontWarnUnrecog);
    baseused = [baseused;tmpBU]; %#ok<AGROW>
  elseif isstruct(sbase.(f)) && ~isstruct(sover.(f))
    warning('structoverlay:badval','Ignoring non-struct value of ''%s''.',...
      newpath);
    baseused{end+1,1} = newpath; %#ok<AGROW>
  else % sbase.(f) is not a struct
    sbase.(f) = sover.(f);
    % sover.(f) might be the same value as sbase.(f), but we don't consider
    % that base is being used in that case
  end
end

s = sbase;
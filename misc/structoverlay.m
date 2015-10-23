function s = structoverlay(sbase,sover,varargin)
% s = structoverlay(sbase,sover,varargin)
% Overlay 'leaf nodes' of sover onto sbase
%
% optional PVs:
% - 'path'. String, defaults to ''. Current struct "path", eg
% .topfield.subfield.

path = myparse(varargin,'path','');

flds = fieldnames(sover);
for f = flds(:)',f=f{1}; %#ok<FXSET>
  newpath = [path '.' f];
  if ~isfield(sbase,f)
    warning('structoverlay:unrecognizedfield','Ignoring unrecognized field ''%s''.',...
      newpath);
  elseif isstruct(sbase.(f)) && isstruct(sover.(f))
    sbase.(f) = structoverlay(sbase.(f),sover.(f),'path',newpath);
  elseif isstruct(sbase.(f)) && ~isstruct(sover.(f))
    warning('structoverlay:badval','Ignoring non-struct value of ''%s''.',...
      newpath);
  else % sbase.(f) is not a struct
    sbase.(f) = sover.(f);
  end
end

s = sbase;
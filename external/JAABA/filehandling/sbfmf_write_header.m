function indexlocptr = ...
  sbfmf_write_header(fp,version,nframes,...
  differencemode,bgcenter,bgstd)

BGTYPE_DARKONLIGHT = 0;
BGTYPE_LIGHTONDARK = 1;
BGTYPE_OTHER = 2;

[nr,nc] = size(bgcenter);

switch lower(differencemode),
  case 'dark flies on a light background',
    differencemode = BGTYPE_DARKONLIGHT;
  case 'light flies on a dark background',
    differencemode = BGTYPE_LIGHTONDARK;
  otherwise,
    differencemode = BGTYPE_OTHER;
end

nbytesver = length(version);
fwrite(fp,nbytesver,'uint32');
fwrite(fp,version,'char');
fwrite(fp,nc,'uint32');
fwrite(fp,nr,'uint32');
fwrite(fp,nframes,'uint32');
fwrite(fp,differencemode,'uint32');
% don't know index loc yet
indexlocptr = ftell(fp);
fwrite(fp,0,'uint32');
fwrite(fp,nframes,'uint32');
% note that these should already be transpose of what matlab uses
fwrite(fp,bgcenter,'double'); 
fwrite(fp,bgstd,'double');

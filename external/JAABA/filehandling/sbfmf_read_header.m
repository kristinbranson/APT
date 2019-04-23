% [nr,nc,nframes,bgcenter,bgstd,frame2file,version,differencemode] = sbfmf_read_header(filename)

function [nr,nc,nframes,bgcenter,bgstd,frame2file,version,differencemode] = ...
  sbfmf_read_header(filename)

fp = fopen( filename, 'r' );
nbytesver = double( fread( fp, 1, 'uint32' ) );
version = fread(fp,nbytesver);
nc = double(fread(fp,1,'uint32'));
nr = double(fread(fp,1,'uint32'));
nframes = double(fread(fp,1,'uint32'));
differencemode = double(fread(fp,1,'uint32'));
indexloc = double(fread(fp,1,'uint64'));
bgcenter = fread(fp,nr*nc,'double');
bgcenter = reshape(bgcenter,[nr,nc]);
bgstd = fread(fp,nr*nc,'double');
bgstd = reshape(bgstd,[nr,nc]);
fseek(fp,indexloc,'bof');
frame2file = fread(fp,nframes,'uint64');
fclose(fp);
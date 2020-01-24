function [data, stamp] = fmf_read_frame( fp, h, w, bytes_per_chunk, data_format )
% [data, stamp] = fmf_read_frame( fp, f_height, f_width,
% bytes_per_chunk, data_format )
%
% reads image data for the next frame in the file pointer FP
%
% DATA is specified by data_format, as below, F_HEIGHT x F_WIDTH
% STAMP is the timestamp for the frame, double format
%
% If DATA_FORMAT == 'MONO8', data is uint8
%
% JAB 7/1/04


datatype = fmf_get_datatype(data_format);

if feof( fp ),
  stamp = 9e9;
  data = zeros( h, w, datatype );
else
  stamp = fread( fp, 1, 'double' );
end

if feof( fp ),
  stamp = 9e9;
  data = zeros( h, w, datatype );
else
  buf = fread( fp, h*w, datatype);
  if size( buf, 2 ) == 0 | size( buf, 1 ) ~= w*h
    stamp = 9e9;
    data = zeros( h, w, datatype );
  else
    data = reshape( buf, w, h )';
  end
  %old: data = reshape( fread( fp, h*w ), w, h )';
end


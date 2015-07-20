function fmf_write_frame( fp, data, stamp, data_format )
% fmf_write_frame( fp, data, stamp, data_format )
%
% writes image data for the next frame in the file pointer FP
%
% FP is a file pointer to fmf file with header already written
% DATA is specified by data_format, as below, F_HEIGHT x F_WIDTH
% STAMP is the timestamp for the frame, double format
% DATA_FORMAT is the format in which each pixel is stored. 
% If DATA_FORMAT == 'MONO8', data is uint8
%
% KMB 04/07


datatype = fmf_get_datatype(data_format);

% write the timestamp
count = fwrite( fp, stamp, 'double');
% write transpose of data
data = data';
count = fwrite( fp, data(:), datatype);



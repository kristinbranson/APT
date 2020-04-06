function datatype = fmf_get_datatype( data_format )

if strcmp(data_format,'MONO8'),
  datatype = 'uint8';
else,
  if strcmp(data_format,'YUV422'),
    datatype = 'uint16';
  else,
    error( 'Only MONO8 supported for now' );
  end
end
function bits_per_pixel = fmf_get_bits_per_pixel( data_format )

if strcmp(data_format,'MONO8'),
  bits_per_pixel = 8;
elseif strcmp(data_format,'YUV422'),
  bits_per_pixel = 16;
else,
  error( 'Only MONO8 supported for now' );
end

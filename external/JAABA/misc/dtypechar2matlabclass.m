function [matlabclass,bytes_per_element] = dtypechar2matlabclass(dtypechar)

switch dtypechar,
  case {'c','s','p'},
    matlabclass = 'char';
    bytes_per_element = 1;
  case 'b',
    matlabclass = 'int8';
    bytes_per_element = 1;
  case 'B',
    matlabclass = 'uint8';
    bytes_per_element = 1;
  case 'h',
    matlabclass = 'int16';
    bytes_per_element = 2;
  case 'H',
    matlabclass = 'uint16';
    bytes_per_element = 2;
  case {'i','l'},
    matlabclass = 'int32';
    bytes_per_element = 4;
  case {'I','L'},
    matlabclass = 'uint32';
    bytes_per_element = 4;
  case 'q',
    matlabclass = 'int64';
    bytes_per_element = 8;
  case 'Q',
    matlabclass = 'uint64';
    bytes_per_element = 8;
  case 'f',
    matlabclass = 'float';
    bytes_per_element = 4;
  case 'd',
    matlabclass = 'double';
    bytes_per_element = 8;
  otherwise
    error('Unknown data type %s',dtypechar);
end


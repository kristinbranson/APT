function [dtypechar,bytes_per_element] = matlabclass2dtypechar(matlabclass)

switch matlabclass,
  case 'char',
    dtypechar = 'c';
    bytes_per_element = 1;
  case 'int8',
    dtypechar = 'b';
    bytes_per_element = 1;
  case 'uint8',
    dtypechar = 'B';
    bytes_per_element = 1;
  case 'int16',
    dtypechar = 'h';
    bytes_per_element = 2;
  case 'uint16',
    dtypechar = 'H';
    bytes_per_element = 2;
  case 'int32',
    dtypechar = 'i';
    bytes_per_element = 4;
  case 'uint32',
    dtypechar = 'I';
    bytes_per_element = 4;
  case 'int64',
    dtypechar = 'q';
    bytes_per_element = 8;
  case 'uint64',
    dtypechar = 'Q';
    bytes_per_element = 8;
  case 'float',
    dtypechar = 'f';
    bytes_per_element = 4;
  case 'double',
    dtypechar = 'd';
    bytes_per_element = 8;
  otherwise
    error('Unknown class2dtype convertsion for %s',matlabclass);
end


function indexloc = ufmf_write_index(fp,keylocs,framelocs)

DICT_START_CHAR = 'd';
ARRAY_START_CHAR = 'a';

indexloc = ftell(fp);

% write a 'd': 1
fwrite(fp,DICT_START_CHAR,'char');

% write the number of key frames
fwrite(fp,numel(keylocs),'uchar');

for j = 1:numel(keylocs),
  
  % write the length of the key name: 2
  l = fread(fp,1,'ushort');
  % read the key name: l
  key = fread(fp,[1,l],'*char');
  % read the next letter to tell if it is an array or another dictionary
  chunktype = fread(fp,1,'*char');
  if chunktype == DICT_START_CHAR,
    % if it's a 'd', then step back one char and read in the dictionary
    % recursively
    fseek(fp,-1,'cof');
    index.(key) = read_dict(fp);
  elseif chunktype == ARRAY_START_CHAR,
    % array
    
    % read in the data type
    dtypechar = fread(fp,1,'*char');
    [matlabclass,bytes_per_element] = dtypechar2matlabclass(dtypechar);
    
    % read in number of bytes
    l = fread(fp,1,'ulong');
    n = l / bytes_per_element;
    if n ~= round(n),
      error('Length in bytes %d is not divisible by bytes per element %d',l,bytes_per_element);
    end
    
    % read in the index array
    [index.(key),ntrue] = fread(fp,n,['*',matlabclass]);
    if ntrue ~= n,
      warning('Could only read %d/%d bytes for array %s of index',n,ntrue,key);
    end
    
  else
    
    error('Error reading dictionary %s. Expected either ''%s'' or ''%s''.',...
      key,DICT_START_CHAR,ARRAY_START_CHAR);
    
  end

end



% write 0 for now
fwrite(fp,0,'uint64'); 
% max box size
fwrite(fp,boxw,'ushort');
fwrite(fp,boxh,'ushort');
% whether it has fixed size patches
fwrite(fp,is_fixed_size,'uchar');
% coding is the encoding of each bit, e.g. MONO8
coding_str_len = length(coding);
fwrite(fp,coding_str_len,'uchar');
fwrite(fp,coding,'char');
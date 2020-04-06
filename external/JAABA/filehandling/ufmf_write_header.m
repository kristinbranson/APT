function [indexlocpr] = ufmf_write_header(fp,boxw,boxh,coding)

version = 4;
is_fixed_size = false;

% ufmf
fwrite(fp,'ufmf','char');
% version
fwrite(fp,version,'uint');
% don't know indexloc yet
indexlocpr = ftell(fp);
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
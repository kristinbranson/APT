function frameloc = ufmf_write_keyframe(fp,stamp,im,keyframe_type)

w = size(im,1);
h = size(im,2);
dtype_char = matlabclass2dtypechar(dtype(im));

frameloc = ftell(fp);

% this is a keyframe
fwrite(fp,0,'uchar');

% write keyframe_type
fwrite(fp,len(keyframe_type),'uchar');
fwrite(fp,keyframe_type,'char');
% write dtype 
fwrite(fp,dtype_char,structtype.c);
% width, height
fwrite(fp,[w,h],'ushort');
% stamp
fwrite(fp,stamp,'double');

% write the data
fwrite(fp,im,dtype(im));

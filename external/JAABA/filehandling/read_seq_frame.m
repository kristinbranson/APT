function [frame,timestamp]=read_seq_frame(seqinfo,frame_num)
% read_seq_frame: function to read in a single frame from a seq file
%
% form:  frame=read_seq_frame(fname_prefix,frame_num)
%
% fname is a string, the filename of the seq file to read a frame from
% e.g. b6_popcage_16_110405_09.58.30.268.seq.
%
% frame_num is the number of the frame to be returned
%
% frame is a matrix of pixel values (uint8), the video frame.
% (usually 768x1024)
%
% required files: r_readseqinfo.m, parsejpg8 (mex file, platform dependent)

frame=parsejpg8(seqinfo.m_strFileName,seqinfo.m_aiSeekPos(frame_num));
timestamp = seqinfo.m_afTimestamp(frame_num);
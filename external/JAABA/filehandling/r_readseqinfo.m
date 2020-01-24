function seqinfo = r_readseqinfo(fname,indx_fname)
% r_readseqinfo: function to read info from a .seq file
%
% form:seqinfo = r_readseqinfo(fname)
%
% modified from shay ohayon's modification of piotr dollar's initial code

strSeqFileName=fname; % (0)

fid = fopen(fname,'r');
fseek(fid,0,'bof');
% first 4 bytes store OxFEED, next 24 store 'Norpix seq  '
fread(fid,1,'uint32');

%if ~(strcmp(sprintf('%X',fread(fid,1,'uint32')),'FEED')) %db ignoring this bit for a moment -- come back to it, hopefully not v. common
    % Attempt to fix SEQ header.
    %fclose(fid);
    %disp('yo')
    %fprintf('Header is corrupted for file %s!\n', fname);
    %resp = input('Do you want to fix the file [Y]/[N]? : ','s');
    %fb = resp(1) == 'Y' || resp(1) == 'y';
    %if fb  
    %    fnFixSeqHeader(strSeqFileName); 
    %    hFileID = fopen(strSeqFileName);
    %    fseek(hFileID,0,'bof');
    %    assert(strcmp(sprintf('%X',fread(hFileID,1,'uint32')),'FEED'));
    %else
    %    strctMovInfo = [];
    %    return;
    %end;
%end;

assert(strcmp(char(fread(fid,10,'uint16'))','Norpix seq')); %#ok<FREAD>
fseek(fid,4,'cof');
% next 8 bytes for version and header size (1024), then 512 for description
vers=fread(fid,1,'int32'); 
assert(fread(fid,1,'uint32')==1024);
fseek(fid,512,'cof');
% read in more strctMovInfo
buff=fread(fid,9,'uint32'); 
assert(buff(8)==0);
fps = fread(fid,1,'float64');
% store movie information in seqinfo struct
seqinfo=struct( 'm_strFileName', fname,...
                     'm_iWidth',buff(1), ...
                     'm_iHeight',buff(2), ...
                     'm_iImageBitDepth',buff(3), ...
                     'm_iImageBitDepthReal',buff(4), ...
                     'm_iImageSizeBytes',buff(5), ...
                     'm_iImageFormat',buff(6), ...
                     'm_iNumFrames',buff(7), ...
                     'm_iTrueImageSize', buff(9),...
                     'm_fFps',fps, ...
                     'm_iSeqiVersion',vers);                
fclose(fid);

% Read in frame indexing info, automatically generate it if it doesn't exist --
[pathstr,name,ext] = fileparts(fname);
if nargin < 2,
  if isunix || ismac
    indx_fname = [pathstr,'/',name,'.mat'];
  else
    if length(pathstr) == 3 && pathstr(3) == '\'
      indx_fname = [pathstr,name,'.mat'];
    else
      indx_fname = [pathstr,'\',name,'.mat'];
    end
  end;
end

%if ~exist(indx_fname,'file')  % don't do this just now--files are way too
%big, takes forever and *should* always be done already before have
%tracking data
%    [aiSeekPos, afTimestamp] = ...
%        fnGenerateSeqSeekInfo(seqinfo,seqinfo.m_iNumFrames);  
%    seqinfo.m_aiSeekPos = aiSeekPos;
%    seqinfo.m_afTimestamp = afTimestamp;    
%    save(indx_fname,'strSeqFileName','aiSeekPos','afTimestamp');  
%else
tmp = load(indx_fname);
seqinfo.m_aiSeekPos = tmp.aiSeekPos;
seqinfo.m_afTimestamp = tmp.afTimestamp;
%end;

return;


%function [aiSeekPos, afTimestamp] = fnGenerateSeqSeekInfo(strctMovInfo, iNumFrames)
%aiSeekPos = zeros(1, iNumFrames);
%afTimestamp = zeros(1, iNumFrames);
%hFileID = fopen(strctMovInfo.m_strFileName);  


% notes
%
% (0)  this is what shay calls it, so have to use all these clunky names for
% consistency, grrr.
function [img,map] = getImageFromFile(filename)
%getImageFromFile retrieves image from file

%   Copyright 2006-2010 The MathWorks, Inc.  
%   $Revision: 1.1.6.12 $  $Date: 2011/11/09 16:50:01 $

if ~ischar(filename)
    error(message('images:getImageFromFile:invalidType'))
end

if ~exist(filename, 'file')
  error(message('images:getImageFromFile:fileDoesNotExist', filename))
end

try
  img_info = [];  % Assign empty, so that it's initialized if imfinfo fails.
  img_info = imfinfo(filename);
  [img,map] = imread(filename);
  if numel(img_info) > 1
      warning(message('images:getImageFromFile:multiframeFile', filename))
  end
  
catch ME
        
    is_tif = ~isempty(img_info) && ...
            isfield(img_info(1),'Format') && ...
            strcmpi(img_info(1).Format,'tif');
        
    % Two different exceptions may be thrown as a result of an out of
    % memory state when reading a TIF file.
    % If rtifc fails in mxCreateNumericArray, MATLAB:nomem is thrown. If rtifc
    % fails in mxCreateUninitNumericArray, then MATLAB:pmaxsize is thrown.
    tif_out_of_memory = is_tif &&...
            ( strcmp(ME.identifier,'MATLAB:nomem') ||...
              strcmp(ME.identifier,'MATLAB:pmaxsize'));
    
    % suggest rsets if they ran out of memory with a tif file    
    if tif_out_of_memory

        outOfMemTifException = MException('images:getImageFromFile:OutOfMemTif',...
            getString(message('images:getImageFromFile:OutOfMemTif')));
        throw(outOfMemTifException);
    end
    
    if (isdicom(filename))
        
        img_info = dicominfo(filename);
        if isfield(img_info,'NumberOfFrames')
            [img,map] = dicomread(img_info,'Frames',1);
            warning(message('images:getImageFromFile:multiframeFile', filename))
        else
            [img,map] = dicomread(img_info);
        end
        
    elseif (isnitf(filename))

        [tf, eid, msg] = iptui.isNitfSupported(filename);
        if (~tf)
            throw(MException(eid, msg));
        end
        
        img = nitfread(filename);
        map = [];
        
    else
        
        % unknown error, re-throw original exception from imfinfo/imread
        rethrow(ME);
        
    end

end    


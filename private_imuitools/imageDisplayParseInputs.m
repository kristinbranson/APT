function [common_args,specific_args] = imageDisplayParseInputs(varargin)
%imageDisplayParseInputs Parse inputs for image display functions.
%   [common_args,specific_args] =
%   imageDisplayParseInputs(specificNames,varargin) parse inputs for image display
%   functions including properties specified in property value pairs. Client-specific
%   arguments specified in specificNames are returned in specific_args.
%
%   common_args is a structure containing arguments that are shared by imtool and imshow.
%
%   specific_args is a structure containing arguments that are specific to a
%   particular client. Arguments specified in specific_args will be returned
%   as fields of specific_args if they are given as p/v pairs by clients.
%
%   Valid syntaxes:
%   DISPLAY_FUNCTION refers to any image display function that uses this utility
%   for input parsing.
%   DISPLAY_FUNCTION - no arguments
%   DISPLAY_FUNCTION(I)
%   DISPLAY_FUNCTION(I,[LOW HIGH])
%   DISPLAY_FUNCTION(RGB)
%   DISPLAY_FUNCTION(BW)
%   DISPLAY_FUNCTION(X,MAP)
%   DISPLAY_FUNCTION(FILENAME)
%
%   DISPLAY_FUNCTION(...,PARAM1,VAL1,PARAM2,VAL2,...)
%   Parameters include:
%      'Colormap', CMAP
%      'DisplayRange', RANGE
%      'InitialMagnification', INITIAL_MAG
%      'XData', X
%      'YData', Y

%   Copyright 1993-2010 The MathWorks, Inc.
%   $Revision: 1.1.8.9 $  $Date: 2011/01/21 19:45:43 $

% I/O Spec
%   I            2-D, real, full matrix of class:
%                uint8, uint16, int16, single, or double.
%
%   BW           2-D, real full matrix of class logical.
%
%   RGB          M-by-N-by-3 3-D real, full array of class:
%                uint8, uint16, int16, single, or double.
%
%   X            2-D, real, full matrix of class:
%                uint8, uint16, double
%                if isa(X,'uint8') || isa(X,'uint16'): X <= size(MAP,1)-1
%                if isa(X,'double'): 1 <= X <= size(MAP,1)
%
%   MAP,CMAP     2-D, real, full matrix
%                if isa(X,'uint8'): size(MAP,1) <= 256
%                if isa(X,'uint16') || isa(X,'double'): size(MAP,1) <= 65536
%
%   RANGE        2 element vector or empty array, double
%
%   FILENAME     String of a valid filename that is on the path.
%                File must be readable by IMREAD or DICOMREAD.
%
%   INITIAL_MAG  'adaptive', 'fit'
%                numeric scalar
%
%   X,Y          2-element vector, can be more than 2 elements but only
%                first and last are used.
%
%   STYLE        'docked', 'normal'
%
%   H            image object (possibly subclass of HG image object with
%                access to more navigational API)

[common_args,specific_args] = imageDisplayParsePVPairs(varargin{:});
filename_specified = ~isempty(common_args.Filename);
if filename_specified
    
    [common_args.CData,common_args.Map] = ...
        getImageFromFile(common_args.Filename);
    
end

common_args = imageDisplayValidateParams(common_args);


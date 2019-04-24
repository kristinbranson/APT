function cmap=morgenstemning(n,varargin)
%MORGENSTEMNING Colormap that increases linearly in lightness (with colors)
%
%	Written by Matthias Geissbuehler - matthias.geissbuehler@a3.epfl.ch
%	January 2013
%
%   Colormap that increases linearly in lightness (such as a pure black to white
%   map) but incorporates additional colors that help to emphasize the
%   transitions and hence enhance the perception of the data.
%   This colormap is designed to be printer-friendly both for color printers as
%   as well as B&W printers.
%
%   Credit: The idea of the passages over blue&red stems from ImageJ's LUT 'Fire'
%   Our colormap corrects the color-printout-problems as well as the
%   non-linearity in the fire-colormap which would make it incompatible
%   with a B&W printing.
%
%
%   See also: isolum, ametrine
%
%
%   Please feel free to use this colormap at your own convenience.
%   A citation to the original article is of course appreciated, however not "mandatory" :-)
%   
%   M. Geissbuehler and T. Lasser "How to display data by color schemes compatible
%   with red-green color perception deficiencies" Opt. Express 21, 9862-9874 (2013)
%   http://www.opticsinfobase.org/oe/abstract.cfm?URI=oe-21-8-9862
%
%
%   For more detailed information, please see:
%   http://lob.epfl.ch -> Research -> Color maps
%
%
%   Usage:
%   cmap = morgenstemning(n)
%
%   All arguments are optional:
%
%   n           The number of elements (256)
%
%   Further on, the following options can be applied
%     'minColor' The absolute minimum value can have a different color
%                ('none'), 'white','black','lightgray', 'darkgray'
%                or any RGB value ex: [0 1 0]
%     'maxColor' The absolute maximum value can have a different color
%     'invert'   (0), 1=invert the whole colormap
%     'gamma'    The gamma of the monitor to be used (1.8)
%
%
%   Examples:
%     figure; imagesc(peaks(200));
%     colormap(morgenstemning)
%     colorbar
%
%     figure; imagesc(peaks(200));
%     colormap(morgenstemning(256,'minColor','black','maxColor',[0 1 0]))
%     colorbar
%
%     figure; imagesc(peaks(200));
%     colormap(morgenstemning(256,'invert',1,'minColor','darkgray'))
%     colorbar
%
%
%
%
% 
%     Copyright (c) 2013, Matthias Geissbuehler
%     All rights reserved.
% 
%     Redistribution and use in source and binary forms, with or without
%     modification, are permitted provided that the following conditions are
%     met:
% 
%         * Redistributions of source code must retain the above copyright
%           notice, this list of conditions and the following disclaimer.
%         * Redistributions in binary form must reproduce the above copyright
%           notice, this list of conditions and the following disclaimer in
%           the documentation and/or other materials provided with the distribution
% 
%     THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
%     AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
%     IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
%     ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
%     LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
%     CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
%     SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
%     INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
%     CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
%     ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
%     POSSIBILITY OF SUCH DAMAGE.

%   Copyright 2013 Matthias Geissbuehler - matthias.geissbuehler@a3.epfl.ch
%   $Revision: 3.0 $  $Date: 2013/01/29 12:00:00 $
p=inputParser;
p.addParamValue('minColor','none');
p.addParamValue('maxColor','none');
p.addParamValue('invert',0, @(x)x==0 || x==1);
p.addParamValue('gamma',1.8, @(x)x>0);

if nargin==1
    p.addRequired('n', @(x)x>0 && mod(x,1)==0);
    p.parse(n);
elseif nargin>1
    p.addRequired('n', @(x)x>0 && mod(x,1)==0);
    p.parse(n, varargin{:});
else
    p.addParamValue('n',256, @(x)x>0 && mod(x,1)==0);
    p.parse();
end
config = p.Results;
n=config.n;

%the ControlPoints
cP(:,1) = [0 0 0]./255;
cP(:,2) = [25 53 95]./255;         %cyan
cP(:,3) = [192 27 111]./255;       %redish-magenta
cP(:,4) = [252 229 0]./255;        %yellow
cP(:,5) = [255 255 255]./255;

number_of_elements_reached = false;
last_n = size(cP,2);
curr_n = last_n .* 2 - 1;
last_cmap = double(cP');

% Normalization and smooth interpolation while keeping
% strictly monotonically increasing gray-values:
%
% 1. interpolate 2x the number of points of the previous cmap (controlpoints)
% 2. normalize all of them
% 3. Loop from 1. until number of points is >n
% 4. Interpolate to the correct number of points (n)

while ~number_of_elements_reached;
    cmap = abs(interp1((1:last_n),last_cmap,linspace(1,last_n,curr_n),'pchip'));  % Interpolation between the control-Points
    
    checkIfAnyAbove1 = 1;
    while checkIfAnyAbove1
        % Normalization by calculation of the gray-value
        % using the average RGB-value (gamma-corrected)
        tempgraymap = mean(cmap.^config.gamma,2);
        tempgraymap = tempgraymap .^(1/config.gamma);
        cmap(:,1)=cmap(:,1)./tempgraymap.*linspace(0,1,curr_n)';
        cmap(:,2)=cmap(:,2)./tempgraymap.*linspace(0,1,curr_n)';
        cmap(:,3)=cmap(:,3)./tempgraymap.*linspace(0,1,curr_n)';
        cmap(isnan(cmap))=0;
        cmap = round(10000*cmap)./10000; % staying within reasonable required precision
        
        % check if during normalization any value is now bigger than 1
        above1 = cmap>1;
        if sum(above1(:))
            mydiff = 0.025;
            if sum(above1(:,1))  % any R>1 ?
                myIndexes = find(above1(:,1));
                cmap(myIndexes,1) = (1-mydiff) .* cmap(myIndexes,1);                          % remove a little bit
                cmap(myIndexes,2) = (mydiff/2) .* (1-cmap(myIndexes,2)) + cmap(myIndexes,2);  % add a little bit to other values
                cmap(myIndexes,3) = (mydiff/2) .* (1-cmap(myIndexes,3)) + cmap(myIndexes,3);  % add a little bit to other values
            end
            if sum(above1(:,2))  % any G>1 ?
                myIndexes = find(above1(:,2));
                cmap(myIndexes,2) = (1-mydiff) .* cmap(myIndexes,2);                          % remove a little bit
                cmap(myIndexes,1) = (mydiff/2) .* (1-cmap(myIndexes,1)) + cmap(myIndexes,1);  % add a little bit to other values
                cmap(myIndexes,3) = (mydiff/2) .* (1-cmap(myIndexes,3)) + cmap(myIndexes,3);  % add a little bit to other values
            end
            if sum(above1(:,3))  % any B>1 ?
                myIndexes = find(above1(:,3));
                cmap(myIndexes,3) = (1-mydiff) .* cmap(myIndexes,3);                          % remove a little bit
                cmap(myIndexes,1) = (mydiff/2) .* (1-cmap(myIndexes,1)) + cmap(myIndexes,1);  % add a little bit to other values
                cmap(myIndexes,2) = (mydiff/2) .* (1-cmap(myIndexes,2)) + cmap(myIndexes,2);  % add a little bit to other values
            end
            checkIfAnyAbove1 = 1;
        else
            checkIfAnyAbove1 = 0;
        end
    end
    last_n = curr_n;
    curr_n = last_n .* 2 - 1;
    last_cmap = cmap;
    if last_n > n
        number_of_elements_reached = true;
    end
end
cmap = abs(interp1((1:last_n),last_cmap,linspace(1,last_n,n)));


% Additional modifications of the colormap
if config.invert
    cmap = flipud(cmap);
end

if ischar(config.minColor)
    if ~strcmp(config.minColor,'none')
        switch config.minColor
            case 'white'
                cmap(1,:) = [1 1 1];
            case 'black'
                cmap(1,:) = [0 0 0];
            case 'lightgray'
                cmap(1,:) = [0.8 0.8 0.8];
            case 'darkgray'
                cmap(1,:) = [0.2 0.2 0.2];
        end
    end
else
    cmap(1,:) = config.minColor;
end
if ischar(config.maxColor)
    if ~strcmp(config.maxColor,'none')
        switch config.maxColor
            case 'white'
                cmap(end,:) = [1 1 1];
            case 'black'
                cmap(end,:) = [0 0 0];
            case 'lightgray'
                cmap(end,:) = [0.8 0.8 0.8];
            case 'darkgray'
                cmap(end,:) = [0.2 0.2 0.2];
        end
    end
else
    cmap(end,:) = config.maxColor;
end

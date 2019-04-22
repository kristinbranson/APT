function cmap=ametrine(n,varargin)
%AMETRINE "Nearly" isoluminant-Colormap compatible with red-green color perception deficiencies
%
%	Written by Matthias Geissbuehler - matthias.geissbuehler@a3.epfl.ch
%	January 2013
%
%   Features:
%     1) All colors have the same luminescence (ideal for lifetime
%        images that will be displayed with an additional transparency map
%        to "mask" places where the lifetime is not well defined)
%     2) Color vision deficient persons can only see reduced color: as much
%        as 10% of adult male persons have a red-green defiency (either
%        Deuteranope  or Protanope) -> as a result they can only distinguish
%        between blue and yellow. A colormap which is "save" for color vision
%        deficient persons is hence only based on these colors.
%        However: people with normal vision DO have a larger space of colors
%        available: it would be a pity to discard this freedom. So the goal
%        must be a colormap that is both using as many colors as possible
%        for normal-sighted people as well as a colormap that will "look"
%        blue-yellow to people with colorblindness without transitions that
%        falsify the information by including a non-distinct transitions
%        (as is the case for many colormaps based on the whole spectrum
%        (ex. rainbow or jet).
%        That's what this colormap here tries to achieve.
%     3) In order to be save for publications, the colormap uses colors that
%        are only from the CMYK colorspace (or at least not too far)
%     4) In comparison to "isolum", this colormap slightly trades off
%        isoluminescence for a higher color contrast
%
%
%   See also: isolum, morgenstemning
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
%   cmap = ametrine(n)
%
%   All arguments are optional:
%
%   n           The number of elements (256)
%
%   Further on, the following options can be applied
%     'gamma'    The gamma of the monitor to be used (1.8)
%     'minColor' The absolute minimum value can have a different color
%                ('none'), 'white','black','lightgray', 'darkgray'
%                or any RGB value ex: [0 1 0]
%     'maxColor' The absolute maximum value can have a different color
%     'invert'   (0), 1=invert the whole colormap
%
%   Examples:
%     figure; imagesc(peaks(200));
%     colormap(ametrine)
%     colorbar
%
%     figure; imagesc(peaks(200));
%     colormap(ametrine(256,'gamma',1.8,'minColor','black','maxColor',[0 1 0]))
%     colorbar
%
%     figure; imagesc(peaks(200));
%     colormap(ametrine(256,'invert',1,'minColor','white'))
%     colorbar
%
%
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
p.addParamValue('gamma',1.8, @(x)x>0);
p.addParamValue('minColor','none');
p.addParamValue('maxColor','none');
p.addParamValue('invert',0, @(x)x==0 || x==1);

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

%the ControlPoints and the spacing between them

%the ControlPoints in a bit more colorful variant -> slightly less
%isoluminescence, but gives a more vivid look
cP(:,1) = [30  60  150]./255; k(1)=1;  %cyan at index 1
cP(:,2) = [180 90  155]./255; k(3)=17; %purple at index 17
cP(:,3) = [230 85  65 ]./255; k(4)=32; %redish at index 32
cP(:,4) = [220 220 0  ]./255; k(5)=64; %yellow at index 64
for i=1:3
    f{i}   = linspace(0,1,(k(i+1)-k(i)+1))';  % linear space between these controlpoints
    ind{i} = linspace(k(i),k(i+1),(k(i+1)-k(i)+1))';
end
cmap = interp1((1:4),cP',linspace(1,4,64)); % for non-iso points, a normal interpolation gives better results


% normal linear interpolation to achieve the required number of points for the colormap
cmap = abs(interp1(linspace(0,1,size(cmap,1)),cmap,linspace(0,1,n)));

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

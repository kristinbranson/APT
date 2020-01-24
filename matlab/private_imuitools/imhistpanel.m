function hout = imhistpanel(parent,himage)
%IMHISTPANEL histogram display panel
%   HOUT = IMHISTPANEL(PARENT,HIMAGE) creates a histogram display panel
%   associated with the image in specified by the handle HIMAGE, called the
%   target image. HPARENT is the handle to the figure or uipanel object that
%   will contain the histogram display panel. 
%
%   This is currently only used by IMCONTRAST.

%   Copyright 2005-2008 The MathWorks, Inc.
%   $Revision: 1.1.6.8 $  $Date: 2010/06/07 16:32:32 $

  histStruct = getHistogramData(himage);
  histRange = histStruct.histRange;
  finalBins = histStruct.finalBins;
  counts    = histStruct.counts;
  
  minX = double(histRange(1));
  maxX = double(histRange(2));
  maxY = max(counts);
  
  hout = uipanel('Parent', parent, ...
                 'Units', 'normalized');
  
  setChildColorToMatchParent(hout,parent);
  
  hAx = axes('Parent', hout);
  
  hStem = stem(hAx, finalBins, counts);  
  set(hStem, 'Marker', 'none')

  set(hAx,'YTick', []);
  set(hAx, 'YLim', [0 maxY]);
  
  xTick = get(hAx, 'XTick');
  xTickSpacing = xTick(2) - xTick(1);

  % Add a little buffer to the Xlim so that you can see the counts at the min and
  % max of the data. Found 5 by experimenting with different images (see testing
  % section in tech ref).
  buffer = xTickSpacing / 5;  
  isDoubleData = strcmp(class(get(himage,'cdata')),'double');
  if ~isDoubleData
     buffer = ceil(buffer);
  end
  Xlim1 = minX - buffer;
  Xlim2 = maxX + buffer;
  set(hAx, 'XLim', [Xlim1 Xlim2]);

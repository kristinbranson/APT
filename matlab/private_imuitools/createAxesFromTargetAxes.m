function haxes = createAxesFromTargetAxes(targetAxes,parent)
%createAxesFromTargetAxes creates axes using target axes.
%   H = createAxesFromTargetAxes(TARGETAXES,PARENT) creates a new image
%   object that has the same properties as the TARGETIMAGE. The image object H
%   is parented to PARENT.
%
%   We created this function in order to avoid using COPYOBJ, which was
%   causing a host of problems in IMPIXELREGIONPANEL and in
%   IMOVERVIEWPANEL.
  
%   Copyright 2007 The MathWorks, Inc.  
%   $Revision: 1.1.6.1 $  $Date: 2007/09/18 02:09:56 $

struct         = getCommonAxesProperties;

struct.Clim    = get(targetAxes,'Clim');
struct.Parent  = parent;
struct.Units   = get(targetAxes,'Units');
struct.Visible = get(targetAxes,'Visible');
struct.XLim    = get(targetAxes,'XLim');
struct.YLim    = get(targetAxes,'YLim');

haxes = axes(struct);


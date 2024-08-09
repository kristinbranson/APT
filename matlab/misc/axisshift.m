function axisshift(ax,xs,ys)

% v = axis(ax);
% v(1:2) = v(1:2)+xs;
% v(3:4) = v(3:4)+ys;
% axis(ax,v);

xl0 = ax.XLim ;
yl0 = ax.YLim ;
xl = xl0 + xs ;
yl = yl0 + ys ;
set(ax, 'XLim', xl, 'YLim', yl) ;  % this is the slowest line by far

% Could we use the CameraView* instead to set the view?  Would that be faster?
% Seems like the sort of thing that might be faster to do manually, rotating
% and zooming the relevant part of the frame using imwarp (or a C++ version of
% imwarp), and rotating and zooming the decorations ourselves.

end

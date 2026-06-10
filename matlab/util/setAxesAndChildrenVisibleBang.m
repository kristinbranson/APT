function setAxesAndChildrenVisibleBang(ax, tfIsVisible)
% Set the visibility of an axes and all its children.

visibleValue = onIff(tfIsVisible) ;
ax.Visible = visibleValue ;
children = ax.Children ;
for i = 1:numel(children)
  if isprop(children(i), 'Visible')
    children(i).Visible = visibleValue ;
  end
end

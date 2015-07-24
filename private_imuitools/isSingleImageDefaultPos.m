function is_single = isSingleImageDefaultPos(fig_handle, ax_handle)
%isSingleImageDefaultPos 
%   IS_SINGLE = isSingleImageDefaultPos(FIG,AX) returns true when it the
%   axes AX is alone in the figure FIG and AX is in its default
%   position. This heuristic means it is safe to resize the figure to
%   accommodate user requests on the initial magnification.

%   Copyright 1993-2003 The MathWorks, Inc.  
%   $Revision: 1.1.8.1 $  $Date: 2004/08/10 01:50:28 $

more_than_one_image = (length(findobj(ax_handle, 'Type', 'image')) > 1);
if more_than_one_image
    is_single = false;

else
    fig_kids = allchild(fig_handle);
    kids = [findobj(fig_kids, 'Type', 'axes') ;
            findobj(fig_kids, 'Type', 'uicontrol', 'Visible', 'on')];
    axes_has_company = (length(kids) > 1);
    
    if axes_has_company
        is_single = false;
    else
        active_property = get(ax_handle,'ActivePositionProperty');
        fig = ancestor(ax_handle,'Figure');
        axes_in_default_pos = (isequal(get(ax_handle, active_property), ...
                                       get(fig, ...
                                           ['DefaultAxes' active_property])));
        if axes_in_default_pos
            is_single = true;            
        else
            is_single = false;
        end
    end
end

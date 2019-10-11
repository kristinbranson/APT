function impan(obj,eventdata) %#ok eventdata needed by HG caller
%IMPAN Interactively pan scrollpanel.

%   Copyright 1993-2009 The MathWorks, Inc.
%   $Revision: 1.1.8.8 $  $Date: 2009/11/09 16:25:29 $

hIm = obj;
hScrollpanel = checkimscrollpanel(hIm,mfilename,'HIMAGE');
apiScrollpanel = iptgetapi(hScrollpanel);

hFig = ancestor(obj, 'figure');

singleClick = strcmp(get(hFig, 'SelectionType'), 'normal');

if singleClick
    % Note: we must use the figure CurrentPoint property to make sure
    % the calculation of the deltas is invariant to the image being dragged.
    start_point = get(hFig, 'CurrentPoint');
    start_x = start_point(1,1,1);
    start_y = start_point(1,2,1);

    start_pos_im = apiScrollpanel.getVisibleImageRect();
    start_pos_im(3:4) = [];

    screenPerImagePixels = apiScrollpanel.getMagnification();

    drag_motion_callback_id = iptaddcallback(hFig, ...
        'WindowButtonMotionFcn', ...
        @dragMotion);

    drag_up_callback_id = iptaddcallback(hFig, ...
        'WindowButtonUpFcn', ...
        @stopDrag);
end

    %----------------------------
    function dragMotion(varargin) % varargin needed by HG caller

        % The closedhand pointer is not provided by the HG figure window. Since we
        % are using setptr, disable the pointer manager to prevent manager
        % from caching hand cursors.
        iptPointerManager(hFig,'Disable');
        setptr(hFig,'closedhand')

        % Note: we must use the figure CurrentPoint property to make sure
        % the calculation of the deltas is invariant to the image being dragged.
        new_point = get(hFig, 'CurrentPoint');
        cp = new_point(1,1:2);
        delta_x_screen = cp(1) - start_x;
        delta_y_screen = cp(2) - start_y;
        
        new_pos_im =  start_pos_im - ...
            [delta_x_screen -delta_y_screen]/screenPerImagePixels;

        new_pos_im = constrainDrag(new_pos_im);
        apiScrollpanel.setVisibleLocation(new_pos_im(1),new_pos_im(2));

        % force update, see: g295731, g301382, g303455
        % drawnow expose 

    end


    %--------------------------
    function stopDrag(varargin) % varargin needed by HG caller

        dragMotion();
        %is_set_position_enabled = true;
        iptremovecallback(hFig, 'WindowButtonMotionFcn', ...
            drag_motion_callback_id);
        iptremovecallback(hFig, 'WindowButtonUpFcn', ...
            drag_up_callback_id);

		setptr(hFig,'hand');
        iptPointerManager(hFig,'Enable');

    end


    %------------------------------------------
    function pos = constrainDrag(proposedPos)

        x = proposedPos(1);
        y = proposedPos(2);

        r = apiScrollpanel.getVisibleImageRect();
        w = r(3);
        h = r(4);
        
        % get image model and cdata sizse
        imModel = getimagemodel(hIm);
        imW = getImageWidth(imModel);
        imH = getImageHeight(imModel);
        
        % get spatial span of data
        xdata = get(hIm,'XData');
        ydata = get(hIm,'YData');
        
        % compute pixel span in each direction
        dx = getDeltaOnePixel(xdata,imW);
        dy = getDeltaOnePixel(ydata,imH);
        
        left_edge   = xdata(1) - dx/2;
        right_edge  = xdata(2) + dx/2;
        top_edge    = ydata(1) - dy/2;
        bottom_edge = ydata(2) + dy/2;
        
        x = min(right_edge  - w, max(x,left_edge) );
        y = min(bottom_edge - h, max(y,top_edge) );
 
        pos = [x y];

    end

end

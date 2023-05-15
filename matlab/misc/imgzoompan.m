function imgzoompan(hfig, varargin)
% imgzoompan provides instant mouse zoom and pan
%
% function imgzoompan(hfig, varargin)
%
%% Purpose
% This function provides instant mouse zoom (mouse wheel) and pan (mouse drag) capabilities 
% to figures, designed for displaying 2D images that require lots of drag & zoom. For more
% details see README file.
%
% 
%% Inputs (optional param/value pairs)
% The following relate to zoom config
% * 'Magnify' General magnitication factor. 1.0 or greater (default: 1.1). A value of 2.0 
%             solves the zoom & pan deformations caused by MATLAB's embedded image resize method.
% * 'XMagnify'        Magnification factor of X axis (default: 1.0).
% * 'YMagnify'        Magnification factor of Y axis (default: 1.0).
% * 'ChangeMagnify'.  Relative increase of the magnification factor. 1.0 or greater (default: 1.1).
% * 'IncreaseChange'  Relative increase in the ChangeMagnify factor. 1.0 or greater (default: 1.1).
% * 'MinValue' Sets the minimum value for Magnify, ChangeMagnify and IncreaseChange (default: 1.1).
% * 'MaxZoomScrollCount' Maximum number of scroll zoom-in steps; might need adjustements depending 
%                        on your image dimensions & Magnify value (default: 30).
% The following relate to pan configuration:
% 'ImgWidth' Original image pixel width. A value of 0 disables the functionality that prevents the 
%            user from dragging and zooming outside of the image (default: 0).
% 'ImgHeight' Original image pixel height (default: 0).
%
%
%% Outputs
%  none
%
% 
%% ACKNOWLEDGEMENTS:
%
% *) Hugo Eyherabide (Hugo.Eyherabide@cs.helsinki.fi) as this project uses his code
%    (FileExchange: zoom_wheel) as reference for zooming functionality.
% *) E. Meade Spratley for his mouse panning example (FileExchange: MousePanningExample).
% *) Alex Burden for his technical and emotional support.
%
% Send code updates, bug reports and comments to: Dany Cabrera (dcabrera@uvic.ca)
% Please visit https://github.com/danyalejandro/imgzoompan (or check the README.md text file) for
% full instructions and examples on how to use this plugin.
%
%% Copyright (c) 2018, Dany Alejandro Cabrera Vargas, University of Victoria, Canada,
% published under BSD license (http://www.opensource.org/licenses/bsd-license.php).
%
% Modified by Allen Lee, 2021


    %  Run in current figure unless otherwise requested
    if isempty(findobj('type','figure'))
        fprintf('%s -- finds no open figure windows. Quitting.\n', mfilename)
        return
    end

    if nargin==0 || isempty(hfig) || ~isa(hfig,'matlab.ui.Figure')
        hfig = gcf;
    end

    % Parse configuration options
    p = inputParser;
    % Zoom configuration options
    p.addOptional('Magnify', 1.1, @isnumeric);
    p.addOptional('XMagnify', 1.0, @isnumeric);
    p.addOptional('YMagnify', 1.0, @isnumeric);
    p.addOptional('ChangeMagnify', 1.1, @isnumeric);
    p.addOptional('IncreaseChange', 1.1, @isnumeric);
    p.addOptional('MinValue', 1.1, @isnumeric);
    p.addOptional('MaxZoomScrollCount', 30, @isnumeric);

    % Pan configuration options
    % adding xmin and ymin for when images are cropped. MK 20230425
    p.addOptional('ImgXMin', 0, @isnumeric);
    p.addOptional('ImgYMin', 0, @isnumeric);
    p.addOptional('ImgWidth', 0, @isnumeric);
    p.addOptional('ImgHeight', 0, @isnumeric);

    % Mouse options and callbacks
    p.addOptional('PanMouseButton', 2, @isnumeric);
    p.addOptional('ResetMouseButton', 3, @isnumeric);
    p.addOptional('ButtonDownFcn',  @(~,~) 0);
    p.addOptional('ButtonUpFcn', @(~,~) 0) ;
    p.addOptional('wbmf',  @(~,~) 0);
    p.addOptional('wbuf', @(~,~) 0) ;
    

    % Parse & Sanitize options
    parse(p, varargin{:});
    opt = p.Results;

    if opt.Magnify<opt.MinValue
        opt.Magnify=opt.MinValue;
    end
    if opt.ChangeMagnify<opt.MinValue
        opt.ChangeMagnify=opt.MinValue;
    end
    if opt.IncreaseChange<opt.MinValue
        opt.IncreaseChange=opt.MinValue;
    end



    % Set up callback functions
    %set(hfig, 'WindowScrollWheelFcn', @zoom_fcn);
    set(hfig, 'WindowButtonDownFcn', @down_fcn);
    %set(hfig, 'WindowButtonUpFcn', @up_fcn);

    zoomScrollCount = 0;
    orig.h=[];
    orig.XLim=[];
    orig.YLim=[];



    % -------------------------------
    % Nested callback functions, etc, follow


    % Applies zoom
    function zoom_fcn(src, cbdata)
        scrollChange = cbdata.VerticalScrollCount; % -1: zoomIn, 1: zoomOut

        if ((zoomScrollCount - scrollChange) <= opt.MaxZoomScrollCount)
            axish = gca;

            if (isempty(orig.h) || axish ~= orig.h)
                orig.h = axish;
                orig.XLim = axish.XLim;
                orig.YLim = axish.YLim;
            end

            % calculate the new XLim and YLim
            cpaxes = mean(axish.CurrentPoint);
            newXLim = (axish.XLim - cpaxes(1)) * (opt.Magnify * opt.XMagnify)^scrollChange + cpaxes(1);
            newYLim = (axish.YLim - cpaxes(2)) * (opt.Magnify * opt.YMagnify)^scrollChange + cpaxes(2);

            newXLim = floor(newXLim);
            newYLim = floor(newYLim);
            % only check for image border location if user provided ImgWidth
            if (opt.ImgWidth > 0)
                if (newXLim(1) >= opt.ImgXMin && newXLim(2) <= opt.ImgWidth && newYLim(1) >= opt.ImgYMin && newYLim(2) <= opt.ImgHeight)
                    axish.XLim = newXLim;
                    axish.YLim = newYLim;
                    zoomScrollCount = zoomScrollCount - scrollChange;
                else
                    axish.XLim = orig.XLim;
                    axish.YLim = orig.YLim;
                    zoomScrollCount = 0;
                end
            else
                axish.XLim = newXLim;
                axish.YLim = newYLim;
                zoomScrollCount = zoomScrollCount - scrollChange;
            end
            %fprintf('XLim: [%.3f, %.3f], YLim: [%.3f, %.3f]\n', axish.XLim(1), axish.XLim(2), axish.YLim(1), axish.YLim(2));
        end
    end %zoom_fcn

    %% Mouse Button Callbacks

    function clickType = buttonNumToName(bt)
      if bt == 1
        clickType = 'normal';
      elseif bt == 2
        clickType = 'alt';
      elseif bt == 3,
        clickType = 'extend';
      end
    end

    function down_fcn(hObj, evt)
        %disp('down_fcn');
        
        opt.ButtonDownFcn(hObj, evt); % First, run callback from options

        clickType = evt.Source.SelectionType;

        % Panning action
        panBt = opt.PanMouseButton;
        if (panBt > 0) && strcmp(buttonNumToName(panBt),clickType),
          guiArea = hittest(hObj);
          parentAxes = ancestor(guiArea,'axes');

          % if the mouse is over the desired axis, trigger the pan fcn
          if ~isempty(parentAxes)
            startPan(parentAxes)
          else
            setptr(evt.Source,'forbidden')
          end
        end
    end %down_fcn

    % Main mouseButtonUp callback
    function up_fcn(hObj, evt)
        opt.ButtonUpFcn(hObj, evt); % First, run callback from options
        
        % Reset action
        clickType = evt.Source.SelectionType;
        resBt = opt.ResetMouseButton;

        if (resBt > 0 && ~isempty(orig.XLim))
          if strcmp(buttonNumToName(resBt),clickType),
            guiArea = hittest(hObj);
            parentAxes = ancestor(guiArea,'axes');
            parentAxes.XLim=orig.XLim;
            parentAxes.YLim=orig.YLim;
          end
        end

        stopPan

    end %up_fcn


    %% AXIS PANNING FUNCTIONS

    % Call this Fcn in your 'WindowButtonDownFcn'
    % Take in desired Axis to pan
    % Get seed points & assign the Panning Fcn to top level Fig
    function startPan(hAx)
        hFig = ancestor(hAx, 'Figure', 'toplevel');   % Parent Fig

        seedPt = get(hAx, 'CurrentPoint'); % Get init mouse position
        seedPt = seedPt(1, :); % Keep only 1st point

        % Temporarily stop 'auto resizing'
        hAx.XLimMode = 'manual'; 
        hAx.YLimMode = 'manual';

        set(hFig,'WindowButtonMotionFcn',{@panningFcn,hAx,seedPt});
        set(hfig, 'WindowButtonUpFcn', @up_fcn);
        setptr(hFig, 'hand'); % Assign 'Panning' cursor
    end %startPan


    % Call this Fcn in your 'WindowButtonUpFcn'
    function stopPan
        set(gcbf,'WindowButtonMotionFcn',opt.wbmf);
        set(hfig, 'WindowButtonUpFcn',opt.wbuf);
        setptr(gcbf,'arrow');
    end %stopPan


    % Controls the real-time panning on the desired axis
    function panningFcn(~,~,hAx,seedPt)
        % Get current mouse position
        currPt = get(hAx,'CurrentPoint');

        % Current Limits [absolute vals]
        XLim = hAx.XLim;
        YLim = hAx.YLim;

        % Original (seed) and Current mouse positions [relative (%) to axes]
        x_seed = (seedPt(1)-XLim(1))/(XLim(2)-XLim(1));
        y_seed = (seedPt(2)-YLim(1))/(YLim(2)-YLim(1));

        x_curr = (currPt(1,1)-XLim(1))/(XLim(2)-XLim(1));
        y_curr = (currPt(1,2)-YLim(1))/(YLim(2)-YLim(1));

        % Change in mouse position [delta relative (%) to axes]
        deltaX = x_curr-x_seed;
        deltaY = y_curr-y_seed;

        % Calculate new axis limits based on mouse position change
        newXLims(1) = -deltaX*diff(XLim)+XLim(1);
        newXLims(2) = newXLims(1)+diff(XLim);

        newYLims(1) = -deltaY*diff(YLim)+YLim(1);
        newYLims(2) = newYLims(1)+diff(YLim);

        % MATLAB lack of anti-aliasing deforms the image if XLims & YLims are not integers
        newXLims = round(newXLims);
        newYLims = round(newYLims);
        if newXLims(1) <= opt.ImgXMin
          newXLims(2) = (newXLims(2)-newXLims(1))+opt.ImgXMin;
          newXLims(1) = opt.ImgXMin;
        elseif newXLims(2) >= opt.ImgWidth
          newXLims(1) = opt.ImgWidth - (newXLims(2)-newXLims(1));
          newXLims(2) = opt.ImgWidth;
        end
        if newYLims(1) <= opt.ImgYMin
          newYLims(2) = (newYLims(2)-newYLims(1))+opt.ImgYMin;
          newYLims(1) = opt.ImgYMin;
        elseif newYLims(2) >= opt.ImgHeight
          newYLims(1) = opt.ImgHeight- (newYLims(2)-newYLims(1));
          newYLims(2) = opt.ImgHeight;
        end
%         set(hAx,'Xlim',newXLims);
%         set(hAx,'Ylim',newYLims);

        % Update Axes limits
        if (newXLims(1) > opt.ImgXMin && newXLims(2) < opt.ImgWidth)
            set(hAx,'Xlim',newXLims);
        end
        if (newYLims(1) > opt.ImgYMin && newYLims(2) < opt.ImgHeight)
            set(hAx,'Ylim',newYLims);
        end
    end %panningFcn

end %imgzoompan

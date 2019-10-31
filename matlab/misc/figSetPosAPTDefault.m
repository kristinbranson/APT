function figSetPosAPTDefault(hFig)
% figSetPosAPTDefault(hFig)
%
% Default position is centered on primary display, with original size if 
% primary display is big enough to accomodate; otherwise smaller/scaled to
% primary display size.

gr = groot;

units0 = gr.Units;
gr.Units = 'pixels';
scszPx = gr.ScreenSize;
gr.Units = units0;
scWidth = scszPx(3);
scHeight = scszPx(4);

units0 = hFig.Units;
oc = onCleanup(@()set(hFig,'Units',units0));
hFig.Units = 'pixels';
APTSIZEFAC = 0.8;
posPx = hFig.Position;

% Ensure fig isn't too big
posPx(3) = min(posPx(3),round(scWidth*APTSIZEFAC));
posPx(4) = min(posPx(4),round(scHeight*APTSIZEFAC));
% Center on primary display
posPx(1) = (scWidth-posPx(3))/2;
posPx(2) = (scHeight-posPx(4))/2;
hFig.Position = posPx;

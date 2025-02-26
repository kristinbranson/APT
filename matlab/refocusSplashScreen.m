function refocusSplashScreen(hfigsplash, hmain)
% Center the hfigsplash figure on the hmain figure, and make hfigsplash the
% current figure, which brings it into the foreground and gives it focus.

if ~isgraphics(hfigsplash),
  return
end

% Get the outer position of the main figure, in pixels
oldunits = get(hmain,'Units');
set(hmain,'Units','pixels');
pos0 = get(hmain,'OuterPosition');
set(hmain,'Units',oldunits);

% Center the splash figure on the main figure
center = pos0([1,2])+pos0([3,4])/2;
pos1 = get(hfigsplash,'Position');
pos2 = [center-pos1(3:4)/2,pos1(3:4)];
set(hfigsplash,'Position',pos2);

% Bring the splash figure to the front, and give it focus
figure(hfigsplash);
drawnow();

end
function deleteValidGraphicsHandles(h)
% Delete both graphics handles and instances of handle classes

% Delete the handle graphics objects
is_graphics = isgraphics(h) ;  % isgraphics() is true for *valid* graphics object handles
h_graphics = h(is_graphics) ;
delete(h_graphics) ;

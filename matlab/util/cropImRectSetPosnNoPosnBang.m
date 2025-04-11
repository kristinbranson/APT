function cropImRectSetPosnNoPosnBang(hRect, pos)
  % Set the hRect's graphics position without triggering its
  % PositionCallback. Works in concert with cbkCropPosn
  tfSetPosnLabeler0 = get(hRect,'UserData');
  set(hRect,'UserData',false);
  hRect.setPosition(pos);
  set(hRect,'UserData',tfSetPosnLabeler0);
end


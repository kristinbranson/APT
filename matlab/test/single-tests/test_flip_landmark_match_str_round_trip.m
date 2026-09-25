function test_flip_landmark_match_str_round_trip()
  % APTParameters.getFlipLandmarkMatchStr() must read back what
  % setFlipLandmarkMatchStr() wrote.
  %
  % Regression test: the getter called APTParameter.getParam(), a class that
  % does not exist (the trailing 's' was missing), so Labeler.setExtraParams()
  % errored whenever it applied a parameter struct.

  parameters = APTParameters.defaultParamsStructAll() ;
  matchString = '1 2, 3 4' ;
  parameters = APTParameters.setFlipLandmarkMatchStr(parameters, matchString) ;
  readBack = APTParameters.getFlipLandmarkMatchStr(parameters) ;
  assert(ischar(readBack) && strcmp(readBack, matchString), ...
         'Flip-landmark match string did not round-trip') ;
end  % function

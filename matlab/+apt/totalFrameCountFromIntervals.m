function result = totalFrameCountFromIntervals(startframes, endframes)
  % Count the total number of frames covered by the intervals in startframes and
  % endframes.  startframes and endframes should be vectors of the same size,
  % startframes(i) giving the first frame index of interval i, and endframes(i) giving the
  % last, both inclusive.  (Frame indices should start at one.)  If either
  % startframes(i) or endframes(i) is <1, that interval is ignored.

  assert(isequal(size(startframes), size(endframes)), 'startframes and endframes must be the same size') ;
  assert(isrow(startframes) || iscolumn(startframes), 'startframes and endframes must be 1d arrays') ;
  is_valid_from_span_index = (startframes>=1) & (endframes>=1) ;
  span_count = sum(is_valid_from_span_index) ;
  first_frame_index_from_span_index = startframes(is_valid_from_span_index) ;
  last_frame_index_from_span_index = endframes(is_valid_from_span_index) ;
  max_frame_index = max(last_frame_index_from_span_index, [], 'all') ;
  is_in_some_span_from_frame_index = false(max_frame_index,1) ;
  for span_index = 1 : span_count
    first_frame_index = first_frame_index_from_span_index(span_index) ;
    last_frame_index = last_frame_index_from_span_index(span_index) ;
    is_in_some_span_from_frame_index(first_frame_index:last_frame_index) = true ;
  end
  result = sum(is_in_some_span_from_frame_index) ;
end

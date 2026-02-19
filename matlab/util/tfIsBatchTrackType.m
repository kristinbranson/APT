function result = tfIsBatchTrackType(track_type)
% Checks if track_type is in fact a valid track type.
% Valid track types are: {'track', 'link', 'detect'}

result = matches(track_type, {'track', 'link', 'detect'}) ;

end

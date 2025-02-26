function result = remove_pair_from_key_value_list_if_present(key_value_list, key_to_remove)
  % Remove the key-value pair matching key_to_remove.  If key is absent, return
  % the key-value list unchanged.
  key_from_pair_index = key_value_list(1:2:end) ;
  value_from_pair_index = key_value_list(2:2:end) ;
  does_match_from_pair_index = strcmp(key_to_remove, key_from_pair_index) ;
  doesnt_match_from_pair_index = ~does_match_from_pair_index ;
  key_from_new_pair_index = key_from_pair_index(doesnt_match_from_pair_index) ;
  value_from_new_pair_index = value_from_pair_index(doesnt_match_from_pair_index) ;
  result_count = length(key_from_new_pair_index) + length(value_from_new_pair_index) ;
  result = cell(1, result_count) ;
  result(1:2:end) = key_from_new_pair_index ;
  result(2:2:end) = value_from_new_pair_index ;
end

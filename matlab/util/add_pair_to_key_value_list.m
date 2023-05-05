function result = add_pair_to_key_value_list(key_value_list, new_key, new_value)
  % Add the given key-value pair to the list, first elimiinating any existing
  % pairs matching new_key in the input list.
  key_from_pair_index = key_value_list(1:2:end) ;
  value_from_pair_index = key_value_list(2:2:end) ;
  does_match_from_pair_index = strcmp(new_key, key_from_pair_index) ;
  key_from_new_pair_index = key_from_pair_index(does_match_from_pair_index) ;
  value_from_new_pair_index = value_from_pair_index(does_match_from_pair_index) ;
  key_from_final_pair_index = horzcat(key_from_new_pair_index, {new_key}) ;
  value_from_final_pair_index = horzcat(value_from_new_pair_index, {new_value}) ;
  result_count = length(key_from_final_pair_index) + length(value_from_final_pair_index) ;
  result = cell(1, result_count) ;
  result(1:2:end) = key_from_final_pair_index ;
  result(2:2:end) = value_from_final_pair_index ;
end

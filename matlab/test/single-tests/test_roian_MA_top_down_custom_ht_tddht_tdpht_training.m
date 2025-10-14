function test_roian_MA_top_down_custom_ht_tddht_tdpht_training()
  % algo_spec = 'ma_top_down_custom_ht' ;
  algo_spec = [ DLNetType.multi_mdn_joint_torch DLNetType.mdn_joint_fpn ] ;
  test_roian_MA_training_helper(algo_spec) ;
end  % function

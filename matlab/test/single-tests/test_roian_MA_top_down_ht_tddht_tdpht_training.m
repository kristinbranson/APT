function test_roian_MA_top_down_ht_tddht_tdpht_training()
  % test_roian_MA_training_helper('ma_top_down_ht_tddht_tdpht') ;
  test_roian_MA_training_helper([ DLNetType.multi_mdn_joint_torch DLNetType.mdn_joint_fpn ]) ;
end  % function

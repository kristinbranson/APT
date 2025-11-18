function test_roian_MA_top_down_custom_bbox_tddobj_tdpobj_training()
  % algo_spec = { 'ma_top_down_custom_bbox', 'detect_mmdetect', 'mdn_joint_fpn' } ;
  algo_spec = [ DLNetType.detect_mmdetect DLNetType.mdn_joint_fpn ]  ;
  test_roian_MA_training_helper(algo_spec) ;
end  % function

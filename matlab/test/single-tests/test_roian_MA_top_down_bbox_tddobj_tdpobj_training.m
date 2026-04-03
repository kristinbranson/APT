function test_roian_MA_top_down_bbox_tddobj_tdpobj_training()
  % test_roian_MA_training_helper('ma_top_down_bbox_tddobj_tdpobj') ;
  test_roian_MA_training_helper([ DLNetType.detect_mmdetect DLNetType.mdn_joint_fpn ]) ;
end  % function

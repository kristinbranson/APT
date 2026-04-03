function test_roian_MA_top_down_bbox_tddobj_tdpobj_ttt()
  % _ttt is short for _training_then_tracking
  % test_roian_MA_training_then_tracking_helper('ma_top_down_bbox_tddobj_tdpobj') ;
  test_roian_MA_training_then_tracking_helper([ DLNetType.detect_mmdetect DLNetType.mdn_joint_fpn ]) ;
end  % function

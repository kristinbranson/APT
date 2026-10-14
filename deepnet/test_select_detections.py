"""Tests for Pose_detect_mmdetect3x.select_detections, APT's post-processing of detector boxes.

  python test_select_detections.py      (or: pytest test_select_detections.py)
"""
import numpy as np

from Pose_detect_mmdetect3x import select_detections, kept_detections


def box(x0, width=100., y0=0., height=100.):
  """Returns an xyxy box."""
  return [x0, y0, x0 + width, y0 + height]


def test_keeps_separate_boxes():
  """Non-overlapping boxes above the score threshold are all kept."""
  boxes = [box(0), box(500), box(1000)]
  keep = select_detections(boxes, np.array([0.9, 0.8, 0.7]), max_count=10, min_count=0)
  assert list(keep) == [0, 1, 2]


def test_suppresses_overlapping_box():
  """A lower-scoring box overlapping a kept box by more than the NMS threshold is dropped."""
  boxes = [box(0), box(5)]   # IoU 0.905
  keep = select_detections(boxes, np.array([0.9, 0.8]), max_count=10, min_count=0)
  assert list(keep) == [0]


def test_suppressed_box_does_not_suppress_others():
  """Only kept boxes suppress: standard NMS.

  B overlaps A (IoU 0.739) and is suppressed. C overlaps B (IoU 0.739) but not A
  (IoU 0.538), so C must survive -- B was discarded and cannot veto it.
  """
  boxes = [box(0), box(15), box(30)]
  keep = select_detections(boxes, np.array([0.9, 0.8, 0.7]), max_count=10, min_count=0)
  assert list(keep) == [0, 2]


def test_score_threshold_and_max_count():
  """Boxes under the score threshold are dropped once min_count are kept; max_count caps the total."""
  boxes = [box(0), box(500), box(1000), box(1500)]
  scores = np.array([0.9, 0.8, 0.1, 0.05])
  assert list(select_detections(boxes, scores, max_count=10, min_count=0)) == [0, 1]
  assert list(select_detections(boxes, scores, max_count=10, min_count=3)) == [0, 1, 2]
  assert list(select_detections(boxes, scores, max_count=1, min_count=0)) == [0]


def test_detr_gets_apt_selection():
  """DETR is NMS-free, so its queries go through APT's NMS and score threshold."""
  boxes = [box(0), box(5), box(500), box(1000)]
  scores = np.array([0.9, 0.8, 0.7, 0.1])
  assert list(kept_detections('detr', boxes, scores, max_count=10, min_count=0)) == [0, 2]


def test_mmdetect_filtered_nets_only_capped():
  """Faster R-CNN and RTMDet already do NMS and score filtering in mmdetection: no second pass."""
  boxes = [box(0), box(5), box(500), box(1000)]
  scores = np.array([0.9, 0.8, 0.7, 0.1])
  for net in ['frcnn', 'rtmdet', 'rtmdet_s']:
    assert list(kept_detections(net, boxes, scores, max_count=10, min_count=0)) == [0, 1, 2, 3]
    assert list(kept_detections(net, boxes, scores, max_count=2, min_count=0)) == [0, 1]


def test_no_boxes():
  """No detections gives no indices."""
  assert len(select_detections(np.zeros((0, 4)), np.zeros(0), max_count=10, min_count=0)) == 0


if __name__ == '__main__':
  tests = [value for name, value in sorted(globals().items()) if name.startswith('test_')]
  failed = 0
  for test in tests:
    try:
      test()
      print(f'ok    {test.__name__}')
    except AssertionError:
      failed += 1
      print(f'FAIL  {test.__name__}')
  raise SystemExit(1 if failed else 0)

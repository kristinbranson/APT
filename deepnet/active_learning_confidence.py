"""
Confidence-driven incremental active-learning experiment, generalized across
datasets.

Idea: train on a small labeled subset, then repeat: run the latest
checkpoint over the full candidate pool, rank pool images by the model's own
lowest predicted per-keypoint confidence (images where it missed an animal
entirely go first), add the lowest-confidence images to the training set,
and retrain. A parallel "random" arm adds the same number of random images
each round instead, as the control -- compare the two arms' accuracy on
eval_json round by round to see whether confidence-guided labeling actually
beats random.

Round sizes are NOT a flat N/round -- see round_sizes() -- they're log-spaced
from round 0 (size0 images, default 100) up to frac_final*pool_size (default
the whole pool) by round n_rounds, since model error falls off roughly as a
power law in training-set size and the pool is heavy-tailed (informative
examples get scarcer as labeling progresses), so a fixed absolute increment
is a huge relative jump early and noise-level late.

This generalizes the ChimpACT setup built in this session and the older
"Using Confidence for labeling" experiment for unmarked mice in
script_inc_exp.py. It intentionally does NOT reproduce that experiment's
ROI-count-matching-to-a-parallel-random-run bookkeeping -- both arms here
just add the same schedule-derived number of images/round, nothing more.

Four strategies (STRATEGIES): 'confidence' and 'random' pick that round's
images as above, training each round from scratch (the pretrained backbone).
'confidence_warm' and 'random_warm' train on the EXACT SAME images each
round as their base strategy -- byte-identical train_TF.json, not an
independently-computed selection -- so the only variable that differs from
the base arm is training initialization: each round's weights load from
that warm arm's own previous-round checkpoint (-model_files) instead of
training from scratch (see _train_cmd_args's docstring for exactly what that
does and doesn't carry over). This isolates "does warm-starting help",
holding the data sequence fixed. Because of this, a '_warm' arm's round N
needs its base strategy's round N data to already exist (see
_copy_round_from_base) -- run the base strategy's loop first, far enough
ahead (or to completion), before starting the '_warm' one.

Two environments, one file
---------------------------
`loop` (the unattended driver) runs on the bare host: it only ever needs
os/re/glob/json/subprocess, and submits/polls LSF via `ssh
login1.int.janelia.org bsub|bjobs`, same mechanism PoseTools.submit_job /
get_job_status use in script_inc_exp.py -- reimplemented standalone here
(below) because PoseTools.py imports tensorflow/cv2 at module level, which
aren't installed on the bare host, and `ssh` itself isn't installed inside
the APT singularity container, so neither the original PoseTools functions
nor a bare "import APT_interface" work in the same process as bsub/bjobs.

Anything that needs the model (building a confidence-ranked round, or
evaluating a checkpoint) only ever runs *inside* the container -- either you
invoke it there by hand, or `loop` submits it as (part of) an LSF job that
runs inside the container itself.

Usage
-----
On the bare host, hands off, resumable (kill and rerun any time):
    python active_learning_confidence.py chimpact loop --strategy confidence --n_rounds 8
    python active_learning_confidence.py chimpact loop --strategy confidence_warm --n_rounds 8
    python active_learning_confidence.py chimpact loop --strategy random --n_rounds 8
    python active_learning_confidence.py chimpact loop --strategy random_warm --n_rounds 8
Run one arm's loop at a time; start one before the rest so round 0 (shared
across all four) isn't submitted twice.

Inside the APT singularity container, one round at a time by hand:
    python active_learning_confidence.py chimpact init
    python active_learning_confidence.py chimpact round --round 1 --strategy confidence
    python active_learning_confidence.py chimpact eval --round 1 --strategy confidence

To point this at a new dataset, add an entry to DATASETS below.
"""
import os
import re
import time
import glob
import json
import copy
import argparse
import subprocess
from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np


# ---------------------------------------------------------------------------
# dataset registry -- add a new dataset by adding an entry here
# ---------------------------------------------------------------------------

@dataclass
class Dataset:
  name: str
  project_json: str        # APT project/config snapshot passed to create_conf
  pool_json: str            # full candidate-pool coco json (labels are already
                             # known in it, but only images "added" so far are
                             # used to train -- simulates not having them yet)
  eval_json: str            # held-out set used to score each round
  coco_im_dir: str          # base dir joined with each image's file_name
                             # (harmless if file_name is already absolute)
  base_dir: str              # durable cache root for this experiment
  proj_name: str              # ProjName from project_json -- APT_interface
                               # nests conf.cachedir one level deeper than the
                               # -cache CLI arg: <cache>/<proj_name>/<net_type>/
                               # view_0/<name>/ -- see train_cache_dir()
  net_type: str = 'multi_mdn_joint_torch'
  train_name: str = 'deepnet'
  miss_dist_px: float = 50.0   # mean matched keypoint distance above which an
                                 # animal counts as "missed" -- eyeball a few
                                 # examples for your dataset's image scale
                                 # before trusting this default
  conf_overrides: dict = field(default_factory=dict)   # extra conf_params
                                                          # applied every round


_CHIMPACT_DATA = '/groups/branson/bransonlab/mayank/data/ChimpAct'
_CHIMPACT_BASE = '/groups/branson/bransonlab/mayank/apt_cache_2/chimpact_inc'
_MICE_BASE = '/groups/branson/bransonlab/mayank/apt_cache_2/unmarked_mice_inc'

DATASETS = {
    'chimpact': Dataset(
        name='chimpact',
        project_json=os.path.join(
            _CHIMPACT_DATA,
            'lbl_config_hrformer_20260731/chimpAct/20260731T110421_20260731T111351.json'),
        pool_json=os.path.join(_CHIMPACT_DATA, 'ChimpACT_train_coco.json'),
        eval_json=os.path.join(_CHIMPACT_DATA, 'ChimpACT_val_coco.json'),
        coco_im_dir=os.path.join(_CHIMPACT_BASE, 'frames'),
        base_dir=os.path.join(_CHIMPACT_BASE, 'al'),
        proj_name='chimpAct',
        miss_dist_px=50.0,
        # full res (736x1280) is too slow to iterate on across 8 rounds --
        # halve it. miss_dist_px doesn't need adjusting: predictions are
        # mapped back to original-image pixel space regardless of the
        # network's internal working resolution.
        conf_overrides={'rescale': 2.0},
    ),
    'unmarked_mice': Dataset(
        name='unmarked_mice',
        project_json=os.path.join(
            _MICE_BASE, 'interactive/20241231T012358_20241231T012358.json'),
        pool_json=os.path.join(_MICE_BASE, 'confidence/full_train.json'),
        # gt/comparison's own train_TF_fixed.json was overwritten at some
        # point -- comparison.old/ still has the path-fixed version (see
        # script_inc_exp.py:662-684 for how it was built, if you need to
        # regenerate it against a newer gt/comparison/train_TF.json).
        eval_json=os.path.join(_MICE_BASE, 'gt/comparison.old/train_TF_fixed.json'),
        coco_im_dir='',   # file_name is already absolute in this pool
        base_dir=os.path.join(_MICE_BASE, 'al'),
        proj_name='unmarkedMice',
        miss_dist_px=37.5,   # matches the threshold script_inc_exp.py used (75/2)
    ),
}


# ---------------------------------------------------------------------------
# light helpers -- no APT/tensorflow/cv2 imports, safe on the bare host
# ---------------------------------------------------------------------------

_DEEPNET_DIR = '/groups/branson/bransonlab/mayank/APT_develop/deepnet'
_SIF_DEFAULT = os.path.join(
    _DEEPNET_DIR, 'scripts/apt-20260730-tf215-pytorch21-hopper/apt-20260730-tf215-pytorch21-hopper.sif')
_TORCH_HOME = '/groups/branson/home/kabram/.apt/torch'


def load_json(path):
  """Read a json file."""
  with open(path) as f:
    return json.load(f)


def write_json(obj, path):
  """Write a json file, creating its parent dir if needed."""
  os.makedirs(os.path.dirname(path), exist_ok=True)
  with open(path, 'w') as f:
    json.dump(obj, f)


def checkpoints(cacheDir, trainName='deepnet'):
  """Sorted list of (step, model_file_prefix) checkpoints in cacheDir."""
  files = glob.glob(os.path.join(cacheDir, f'{trainName}-[0-9]*'))
  files = [f for f in files if os.path.splitext(f)[1] in ('.index', '')]
  out = []
  for f in files:
    m = re.search(r'-(\d+)(\.index)?$', f)
    if m:
      out.append((int(m.group(1)), os.path.splitext(f)[0]))
  return sorted(set(out))


def latest_checkpoint(cacheDir, trainName='deepnet'):
  """Highest-step checkpoint prefix in cacheDir, or None if none exist yet."""
  ck = checkpoints(cacheDir, trainName)
  return ck[-1][1] if ck else None


def round_done(cacheDir, dlSteps, trainName='deepnet'):
  """True if this round's checkpoint has already reached dlSteps."""
  ck = checkpoints(cacheDir, trainName)
  return bool(ck) and ck[-1][0] >= dlSteps


def round_dir(ds, round_, strategy=None):
  """cache dir for a given round; round 0 has no strategy (shared ancestor)."""
  if round_ == 0:
    return os.path.join(ds.base_dir, 'round_0')
  assert strategy is not None
  return os.path.join(ds.base_dir, f'round_{round_}_{strategy}')


def round_label(round_, strategy=None):
  """name/job-name suffix for a round, matching round_dir's layout."""
  return 'round_0' if round_ == 0 else f'round_{round_}_{strategy}'


def train_cache_dir(ds, round_, strategy=None):
  """The actual conf.cachedir APT_interface uses for this round: nested one
  level deeper than round_dir() (which is the outer -cache CLI argument) --
  <round_dir>/<proj_name>/<net_type>/view_0/<name>. This is where
  APT_interface itself reads/writes train_db.json, checkpoints, and the
  traindata stats pickle -- round_dir() is NOT conf.cachedir. Confusing the
  two broke the first real round-0 submission: gen_train_samples1 crashed
  looking for train_db.json one level deeper than where it had been written.
  """
  return os.path.join(round_dir(ds, round_, strategy), ds.proj_name, ds.net_type,
                       'view_0', round_label(round_, strategy))


STRATEGIES = ['confidence', 'random', 'confidence_warm', 'random_warm']


def strategy_base(strategy):
  """'confidence' or 'random' -- which selection algorithm a strategy uses,
  stripping the '_warm' suffix if present."""
  return strategy[:-len('_warm')] if strategy.endswith('_warm') else strategy


def strategy_is_warm(strategy):
  """True if this strategy initializes each round's training from the
  previous round's own checkpoint instead of from scratch."""
  return strategy.endswith('_warm')


def round_sizes(pool_size, size0=100, n_rounds=8, frac_final=1.0):
  """Target training-set size after each round 0..n_rounds, log-spaced
  between size0 (round 0, an absolute image count) and frac_final*pool_size
  (round n_rounds).

  Scaling-law motivated: model error tends to fall off as a power law in
  training-set size, so a fixed n_add/round becomes a vanishingly small
  *relative* increment once the training set is already large -- round 1
  might double the labeled set, round 8 would barely move it a couple of
  percent. Spacing round sizes geometrically instead keeps each round's
  relative data increase roughly constant, which is the regime scaling laws
  are actually fit in. This also suits a heavy-tailed pool (the easy, highly
  informative examples get used up first, so later rounds need
  proportionally more new data to find enough of what's left) -- and it
  means round 0 should start as small as is still reasonably trainable
  (default 100 images), not at some arbitrary large fraction, or most of the
  schedule ends up bunched in the already-large-dataset regime where
  marginal gains are small anyway.
  """
  size0 = max(1, int(size0))
  sizeFinal = max(size0 + n_rounds, int(round(pool_size * frac_final)))
  sizes = np.round(np.geomspace(size0, sizeFinal, n_rounds + 1)).astype(int).tolist()
  for i in range(1, len(sizes)):   # geomspace can repeat at small ranges/n_rounds
    if sizes[i] <= sizes[i - 1]:
      sizes[i] = sizes[i - 1] + 1
  return sizes


def _write_round_json(jsonPath, trainCacheDir, data):
  """Write a round's train_TF.json (our own bookkeeping, wherever jsonPath
  says), and a copy at trainCacheDir/train_db.json -- train_db.json has to
  land exactly in conf.cachedir (see train_cache_dir(), NOT round_dir()),
  the filename PoseCommon_pytorch's coco data loader actually reads for
  db_format='coco' (poseConfig.py: trainfilename='train_db' when db_format
  != 'tfrecord'). Our round jsons are already valid coco (images/annotations
  built straight from the pool, itself already coco) -- there's nothing to
  convert, so training is invoked with -skip_db to bypass APT_interface's
  create_coco_db entirely, rather than routing already-coco data through a
  function meant to convert APT's native lbl/locdata format *into* coco
  (which, as of writing, also crashes on already-coco input: it reads a
  T['kpt_info'] key nothing in this codebase ever writes)."""
  write_json(data, jsonPath)
  write_json(data, os.path.join(trainCacheDir, 'train_db.json'))


def build_round0(ds, size0=100, seed=0):
  """Build round 0's train_TF.json: size0 random images from the pool. Pure
  json/numpy -- no GPU, no APT imports. Idempotent."""
  cacheDir = round_dir(ds, 0)
  jsonPath = os.path.join(cacheDir, 'train_TF.json')
  if os.path.exists(jsonPath):
    return jsonPath

  np.random.seed(seed)
  pool = load_json(ds.pool_json)
  nImg = len(pool['images'])
  sel = np.random.choice(nImg, min(int(size0), nImg), replace=False)
  selIds = set(pool['images'][i]['id'] for i in sel)

  round0 = copy.deepcopy(pool)
  round0['images'] = [im for im in pool['images'] if im['id'] in selIds]
  round0['annotations'] = [a for a in pool['annotations'] if a['image_id'] in selIds]
  _write_round_json(jsonPath, train_cache_dir(ds, 0), round0)
  print(f'{ds.name} round 0: {len(round0["images"])} images / '
        f'{len(round0["annotations"])} annotations '
        f'(pool: {len(pool["images"])} / {len(pool["annotations"])})')
  return jsonPath


def _build_round_random(ds, round_, strategy, n_add, seed=None):
  """Build round `round_`'s train_TF.json by adding n_add random images.
  Pure json/numpy -- no GPU, no APT imports. Idempotent. Only ever called
  with the base strategy ('random') -- 'random_warm' reuses this round's
  data verbatim instead (see _copy_round_from_base)."""
  cacheDir = round_dir(ds, round_, strategy)
  jsonPath = os.path.join(cacheDir, 'train_TF.json')
  if os.path.exists(jsonPath):
    return jsonPath

  prevDir = round_dir(ds, round_ - 1, strategy if round_ > 1 else None)
  prevData = load_json(os.path.join(prevDir, 'train_TF.json'))
  usedIds = set(im['id'] for im in prevData['images'])

  pool = load_json(ds.pool_json)
  candidateIdx = [i for i, im in enumerate(pool['images']) if im['id'] not in usedIds]
  assert len(candidateIdx) >= n_add, 'pool exhausted'
  rng = np.random.RandomState(seed if seed is not None else round_)
  chosen = rng.choice(candidateIdx, n_add, replace=False)

  chosenIds = set(pool['images'][i]['id'] for i in chosen)
  newRound = copy.deepcopy(prevData)
  newRound['images'] += [pool['images'][i] for i in chosen]
  newRound['annotations'] += [a for a in pool['annotations'] if a['image_id'] in chosenIds]
  _write_round_json(jsonPath, train_cache_dir(ds, round_, strategy), newRound)
  print(f'{ds.name} round {round_} ({strategy}): added {len(chosen)}, '
        f'total {len(newRound["images"])} images / {len(newRound["annotations"])} annotations')
  return jsonPath


def _copy_round_from_base(ds, round_, strategy):
  """For a '_warm' strategy, round `round_`'s training data is byte-identical
  to its base strategy's (confidence_warm <- confidence, random_warm <-
  random) -- not an independently-made selection. This isolates
  warm-starting as the only variable that differs from the base arm. Pure
  json -- no GPU, no APT imports. Idempotent.

  Requires the base strategy's round `round_` to already be built (its
  training doesn't need to be done yet, just its data selected) -- raises
  with a clear message if not, rather than silently diverging from it.
  """
  base = strategy_base(strategy)
  baseJsonPath = os.path.join(round_dir(ds, round_, base), 'train_TF.json')
  assert os.path.exists(baseJsonPath), (
      f"{strategy}'s round {round_} reuses {base}'s round {round_} data, which doesn't exist yet "
      f"({baseJsonPath}) -- run `loop --strategy {base}` (or `round --strategy {base} --round "
      f"{round_}` by hand) far enough first, then retry {strategy}.")

  cacheDir = round_dir(ds, round_, strategy)
  jsonPath = os.path.join(cacheDir, 'train_TF.json')
  if os.path.exists(jsonPath):
    return jsonPath

  data = load_json(baseJsonPath)
  _write_round_json(jsonPath, train_cache_dir(ds, round_, strategy), data)
  print(f"{ds.name} round {round_} ({strategy}): reusing {base}'s data "
        f'({len(data["images"])} images / {len(data["annotations"])} annotations)')
  return jsonPath


def _train_cmd_args(ds, name, cacheDir, jsonTrnFile, dlSteps, saveStep, maxckpt=20,
                     extraOverrides=None, modelFile=None):
  """APT_interface.py argument string for training this round.

  -conf_params has nargs='*', so it greedily swallows every following token
  that doesn't start with '-' -- including the positional 'train' subcommand
  if nothing '-'-prefixed comes between them (confirmed: this broke the
  first real submission, ate 'train', and left a bare '-use_cache' argparse
  couldn't place). It must be followed immediately by another '-flag', never
  by the bare subcommand. -model_files has the same nargs='*' gotcha, so it
  gets the same treatment.

  modelFile, if given (a checkpoint prefix, e.g. from a warm-start strategy),
  is passed as -model_files: PoseCommon_pytorch.train_wrapper() initializes
  weights from it but still starts the step counter at 0 for this round's
  own dl_steps budget -- it is NOT the same as -continue, which resumes a
  single interrupted run's own step count instead.
  """
  overrides = dict(ds.conf_overrides)
  overrides.update(extraOverrides or {})
  overrides.update({'dl_steps': dlSteps, 'save_step': saveStep, 'maxckpt': maxckpt,
                     'coco_im_dir': ds.coco_im_dir})
  confParamStr = ' '.join(
      f"{k} \"'{v}'\"" if isinstance(v, str) else f'{k} {v}'
      for k, v in overrides.items())
  modelFileArg = f'-model_files {modelFile} ' if modelFile else ''
  # -skip_db, not -use_cache: {cacheDir}/train_db.json is already written
  # (see _write_round_json) with our data, which is already valid coco --
  # -use_cache would route it through create_coco_db's lbl-to-coco
  # conversion, which doesn't apply here (and currently crashes on
  # already-coco input).
  return (f"{ds.project_json} -name {name} -conf_params {confParamStr} "
          f"{modelFileArg}"
          f"-err_file {cacheDir}/train_{name}.err -log_file {cacheDir}/train_{name}.log "
          f"-json_trn_file {jsonTrnFile} -type {ds.net_type} -ignore_local 1 -cache {cacheDir} "
          f"train -skip_db")


# ---------------------------------------------------------------------------
# LSF submission/polling, reimplemented standalone (no PoseTools import) --
# mirrors PoseTools.submit_job / get_job_status, which script_inc_exp.py's
# unmarked-mice rounds used, but that module can't be imported on the bare
# host (imports tensorflow/cv2 at module level), and ssh isn't installed
# inside the singularity container -- so this needs to be its own thing,
# run only on the bare host.
# ---------------------------------------------------------------------------

def _ssh_lsf(cmd, checkOutput=False):
  full = f'''ssh login1.int.janelia.org '. /misc/lsf/conf/profile.lsf; {cmd}' '''
  res = subprocess.run(full, shell=True, capture_output=True, text=True)
  if checkOutput and res.returncode != 0:
    raise RuntimeError(f'{cmd!r} failed: {res.stdout}\n{res.stderr}')
  return res


def get_job_status(jobName):
  """PENDING / RUN / DONE / EXIT / NOT_FOUND for jobName, via bjobs."""
  res = _ssh_lsf(f'bjobs -J {jobName}')
  lines = res.stdout.strip().split('\n')
  if len(lines) < 2 or not lines[1].strip():
    return 'NOT_FOUND'
  status = lines[1].split()[2]
  return 'PENDING' if status == 'PEND' else status


def submit_job(jobName, body, cacheDir, queue='gpu_h100', numcores=5, walltime_min=4320,
               sif=_SIF_DEFAULT):
  """Submit `body` (a bash command string, run inside the APT singularity
  container) as an LSF job named jobName, logging to cacheDir."""
  os.makedirs(cacheDir, exist_ok=True)
  singScript = os.path.join(cacheDir, f'{jobName}.sh')
  log = os.path.join(cacheDir, f'{jobName}.log')
  err = os.path.join(cacheDir, f'{jobName}.err')
  with open(singScript, 'w') as f:
    f.write('#!/bin/bash\n')
    f.write(f'cd {_DEEPNET_DIR}\n')
    f.write(f'export TORCH_HOME={_TORCH_HOME}\n')
    f.write('export OMP_NUM_THREADS=2\n')
    f.write(body + '\n')
  os.chmod(singScript, 0o755)

  bsubCmd = (f'bsub -J {jobName} -oo {log} -eo {err} -n{numcores} -W {walltime_min} '
             f'-gpu "num=1" -q {queue} '
             f'"singularity exec --nv -B /groups/branson -B /nrs/branson {sif} {singScript}"')
  res = _ssh_lsf(bsubCmd)
  if res.returncode != 0:
    raise RuntimeError(f'bsub failed for {jobName}: {res.stdout}\n{res.stderr}')
  print(res.stdout.strip() or f'submitted {jobName}')
  return jobName


def wait_for_job(jobName, poll_interval=300):
  """Block until jobName leaves the LSF queue (PENDING/RUN), printing status
  transitions. Returns the final status string."""
  lastStatus = None
  while True:
    status = get_job_status(jobName)
    if status != lastStatus:
      print(f'{time.strftime("%m/%d %H:%M:%S")}: {jobName} -> {status}', flush=True)
      lastStatus = status
    if status not in ('PENDING', 'RUN'):
      return status
    time.sleep(poll_interval)


# ---------------------------------------------------------------------------
# unattended driver -- bare host only, resumable
# ---------------------------------------------------------------------------

def cmd_loop(ds, strategy, n_rounds, dl_steps=40000, save_step=20000,
             maxckpt=20, size0=100, frac_final=1.0, seed=0, poll_interval=300,
             queue='gpu_h100', walltime_min=4320, lr_drop_step=None):
  """Drive rounds 0..n_rounds for one arm (strategy) end to end: submit
  whatever isn't already finished or queued, wait for it, build the next
  round's selection, submit that, and repeat.

  Round sizes are log-spaced (see round_sizes) between size0 images at
  round 0 and frac_final*pool_size at round n_rounds -- not a flat n_add
  per round.

  Safe to kill (Ctrl-C, terminal closed, machine restart) and rerun: every
  round's "is it done" check is a checkpoint-file test and every "is it
  queued" check is a fresh get_job_status(jobName) call against LSF, so a
  rerun just re-derives current state from disk/the queue instead of
  tracking its own separate progress file.

  For a 'random' round, the next round's train_TF.json is built right here
  (pure json/numpy, no GPU) before submitting a training-only job. For a
  'confidence' round, building it needs a GPU inference pass over the pool,
  so that step is folded into the submitted job itself (this script's own
  `round` subcommand runs first inside the container, then APT_interface.py
  trains). A '_warm' round's data is just a copy of its base strategy's
  round (see _copy_round_from_base) -- always light, regardless of base --
  so nothing GPU-bound ever runs on the bare host that drives this loop.

  Round 0's cache dir is shared across ALL FOUR strategies (it's always
  trained from scratch, nothing to warm-start from yet) -- if you're running
  more than one arm, start one first and let it clear round 0 before
  starting the rest, to avoid a double submit.

  strategy is one of STRATEGIES. The '_warm' variants ('confidence_warm',
  'random_warm') train on the EXACT SAME images each round as their base
  strategy -- not an independently-made selection, see
  _copy_round_from_base -- so a '_warm' arm's round N needs its base
  strategy's round N to already be built; run the base strategy's loop
  first, far enough ahead (or to completion). The only variable that then
  differs from the base arm is training initialization: each round's model
  loads weights from this warm arm's own previous-round checkpoint
  (-model_files) instead of training from scratch -- see
  _train_cmd_args's docstring for exactly what that does and doesn't carry
  over (weights yes, step counter no: still dl_steps fresh steps/round).

  lr_drop_step, if given, overrides PoseCommon_pytorch's lr_drop_step conf
  param (default 0.15) for every round this call submits. That param is the
  fraction of dl_steps, counting from the end, spent at a reduced LR (the
  rest trains at the full base LR) -- since -model_files only transplants
  weights, not the optimizer or step counter (see above), a '_warm' round
  otherwise starts its already-converged weights right back at full LR for
  85% of a fresh dl_steps budget, which is tuned for bootstrapping from a
  generic pretrained backbone, not for gently refining an already
  chimpact-converged model. Pass something close to 1.0 (e.g. 0.95) for a
  '_warm' strategy to spend nearly the whole round at the reduced LR
  instead, and test whether warm-starting helps once it isn't immediately
  fighting its own schedule.
  """
  assert strategy in STRATEGIES, f'strategy must be one of {STRATEGIES}'
  isWarm = strategy_is_warm(strategy)
  baseStrategy = strategy_base(strategy)

  pool = load_json(ds.pool_json)
  sizes = round_sizes(len(pool['images']), size0=size0, n_rounds=n_rounds, frac_final=frac_final)
  print(f'{ds.name} ({strategy}) round sizes (log-spaced): {sizes}')
  trainOverrides = {'lr_drop_step': lr_drop_step} if lr_drop_step is not None else None
  if trainOverrides:
    print(f'{ds.name} ({strategy}): overriding lr_drop_step={lr_drop_step} '
          f'(default 0.15) for every round this call submits')

  for round_ in range(0, n_rounds + 1):
    thisStrategy = strategy if round_ > 0 else None
    cacheDir = round_dir(ds, round_, thisStrategy)
    trainCacheDir = train_cache_dir(ds, round_, thisStrategy)
    jobName = f'{ds.name}_{round_label(round_, thisStrategy)}'
    n_add = sizes[round_] - sizes[round_ - 1] if round_ > 0 else None

    if round_done(trainCacheDir, dl_steps, ds.train_name):
      print(f'round {round_} ({strategy}): already done, skipping')
      continue

    status = get_job_status(jobName)
    if status in ('PENDING', 'RUN'):
      print(f'round {round_} ({strategy}): {jobName} already queued ({status}), waiting')
    else:
      modelFile = None
      if round_ > 0 and isWarm:
        prevTrainCacheDir = train_cache_dir(ds, round_ - 1, strategy if round_ > 1 else None)
        modelFile = latest_checkpoint(prevTrainCacheDir, ds.train_name)
        assert modelFile is not None, (
            f'warm-start round {round_} ({strategy}) needs a checkpoint in '
            f'{prevTrainCacheDir}, found none')

      if round_ == 0:
        jsonPath = build_round0(ds, size0=size0, seed=seed)
        args = _train_cmd_args(ds, round_label(0), cacheDir, jsonPath, dl_steps, save_step, maxckpt,
                                extraOverrides=trainOverrides)
        body = f'python {_DEEPNET_DIR}/APT_interface.py {args}'
      elif isWarm:
        # reuses the base strategy's already-built data -- pure file copy,
        # no GPU needed regardless of whether the base is random or confidence.
        jsonPath = _copy_round_from_base(ds, round_, strategy)
        args = _train_cmd_args(ds, round_label(round_, strategy), cacheDir, jsonPath,
                                dl_steps, save_step, maxckpt, extraOverrides=trainOverrides,
                                modelFile=modelFile)
        body = f'python {_DEEPNET_DIR}/APT_interface.py {args}'
      elif baseStrategy == 'random':
        jsonPath = _build_round_random(ds, round_, strategy, n_add=n_add, seed=seed)
        args = _train_cmd_args(ds, round_label(round_, strategy), cacheDir, jsonPath,
                                dl_steps, save_step, maxckpt, extraOverrides=trainOverrides)
        body = f'python {_DEEPNET_DIR}/APT_interface.py {args}'
      else:
        # confidence selection needs a GPU inference pass -- run it (via this
        # same script's `round` subcommand) as the first step of the job
        # itself, inside the container, then chain straight into training.
        jsonPath = os.path.join(cacheDir, 'train_TF.json')
        args = _train_cmd_args(ds, round_label(round_, strategy), cacheDir, jsonPath,
                                dl_steps, save_step, maxckpt, extraOverrides=trainOverrides)
        body = (f'python {_DEEPNET_DIR}/active_learning_confidence.py {ds.name} round '
                f'--round {round_} --strategy {strategy} --n_add {n_add} '
                f'{"--seed " + str(seed) if seed is not None else ""} && '
                f'python {_DEEPNET_DIR}/APT_interface.py {args}')
      submit_job(jobName, body, cacheDir, queue=queue, walltime_min=walltime_min)
      print(f'round {round_} ({strategy}): submitted {jobName}'
            + (f' (warm-started from {modelFile})' if modelFile else ''))

    finalStatus = wait_for_job(jobName, poll_interval=poll_interval)
    if not round_done(trainCacheDir, dl_steps, ds.train_name):
      raise RuntimeError(
          f'round {round_} ({strategy}) job {jobName} ended ({finalStatus}) without reaching '
          f'dl_steps={dl_steps} -- check {cacheDir}/{jobName}.err and '
          f'{cacheDir}/train_{round_label(round_, thisStrategy)}.err. Fix the problem and rerun '
          f'this command; it will resubmit just this round.')

  print(f'{ds.name} ({strategy}): all {n_rounds} rounds complete')


# ---------------------------------------------------------------------------
# heavy helpers -- need APT_interface/PoseTools (tensorflow/cv2/torch), only
# importable/callable *inside* the APT singularity container
# ---------------------------------------------------------------------------

def find_dist_match(dd):
  """Per-image optimal (Hungarian) match of predicted to GT animals.

  dd: [n_img, n_pred, n_gt, n_kpts] distances. Returns [n_img, n_gt, n_kpts],
  each GT animal's distances to its best-matched prediction (nan where a GT
  animal has no candidate prediction). Ported from reuse.find_dist_match.
  Needs scipy (lazy import -- only called inside the container).
  """
  from scipy import optimize as scipy_opt
  dout = np.full_like(dd[:, 0, :, :], np.nan)
  yy = np.nanmean(dd, axis=-1)
  for a in range(dd.shape[0]):
    selPred = np.any(~np.isnan(yy[a]), axis=1)
    selGt = np.any(~np.isnan(yy[a]), axis=0)
    cmat = yy[a, selPred][:, selGt]
    if cmat.size == 0:
      continue
    idxPred, idxGt = scipy_opt.linear_sum_assignment(cmat)
    origPred = np.where(selPred)[0][idxPred]
    origGt = np.where(selGt)[0][idxGt]
    for k in range(len(idxPred)):
      dout[a, origGt[k], :] = dd[a, origPred[k], origGt[k], :]
  return dout


def _pool_confidence(ds, conf, poolJson, model_file=None):
  """For every image in poolJson, run conf's checkpoint (latest, unless
  model_file names a specific one -- e.g. to compare checkpoints from the
  same run), Hungarian-match predictions to poolJson's own GT animals, and
  return per GT animal slot: its matched confidence (nan if no finite
  prediction existed anywhere in the frame -- a real padding/no-prediction
  case) and its Hungarian-matched mean keypoint distance (nan if the
  Hungarian assignment gave it nothing). Deciding "missed" from that
  distance is left to the caller (_visibility_weighted_priority /
  _write_animal_diagnostics): a fixed pixel threshold conflates a
  near-perfect prediction on a large/close animal with a genuinely bad one
  on a small/distant animal (checked directly: round 1's candidates ranged
  from a 51px miss that was 7% of its animal's bbox diagonal to a 76px miss
  that was 75% of a smaller animal's), so the caller normalizes by that
  animal's own bbox size instead. gtLocs only carries x/y, not the
  per-keypoint COCO visibility code, so this does NOT weight by visibility
  either -- same reasoning, folded in by the caller. Generalizes
  script_inc_exp.py:3006-3042."""
  import APT_interface as apt

  # batch_size=1 for inference: classify_db2's partial-last-batch handling
  # leaves stale data in unused slots of its pre-allocated buffer; batch_size
  # == 1 takes a cleaner full-replacement code path instead. See cmd_eval's
  # matching comment.
  conf.batch_size = 1
  res = apt.classify_db_all(ds.net_type, conf, poolJson, model_file=model_file, name=ds.train_name)
  predLocs, predConf = res[0]['locs'], res[0]['conf']
  gtLocs = res[1]
  info = np.array(res[2], dtype=object)[:, :2]

  dd = np.linalg.norm(predLocs[:, :, None] - gtLocs[:, None], axis=-1)
  ddMatched = find_dist_match(dd)   # [n_img, n_gt, n_kpts], Hungarian-assigned

  hasGt = ~np.all(np.isnan(gtLocs[..., 0]), axis=-1)   # [n_img, n_gt], real animal slot vs padding

  with np.errstate(invalid='ignore'):
    meanDistMatched = np.nanmean(ddMatched, axis=-1)   # [n_img, n_gt]; nan where Hungarian assigned nothing
  meanDistMatched = np.where(hasGt, meanDistMatched, np.nan)

  matchedConf = np.full((predLocs.shape[0], gtLocs.shape[1]), np.nan)
  for i in range(predLocs.shape[0]):
    for a in range(gtLocs.shape[1]):
      if not hasGt[i, a]:
        continue
      distToPreds = np.linalg.norm(gtLocs[i, a, None] - predLocs[i], axis=-1).mean(axis=-1)
      if np.all(np.isnan(distToPreds)):
        continue
      best = int(np.nanargmin(distToPreds))
      matchedConf[i, a] = predConf[i, best].mean(axis=-1)

  return matchedConf, meanDistMatched, info


VIS_WEIGHTS = {0: 0.0, 1: 0.5, 2: 1.0}
# COCO per-keypoint visibility code -> training-value weight:
#   0 = not labeled at all -- no supervision signal there regardless of
#       model confidence
#   1 = labeled but occluded -- a usable label, but the annotator estimated
#       it through occlusion, so it's noisier ground truth and the model's
#       difficulty there is less actionable than on a clearly-visible point
#   2 = labeled and clearly visible -- the most reliable signal
# For ChimpACT specifically these are NOT a long tail: v=0/1/2 split
# roughly 7%/45%/48% of all keypoints (checked directly against
# ChimpACT_train_coco.json), so collapsing v=1 into v=2 (or into v=0) would
# throw away real information, not just smooth over a rare edge case.

MISS_WEIGHT = 1.2
# badness for an animal the model missed outright (no Hungarian-assigned
# prediction close enough -- see _pool_confidence). A merely-low-confidence
# animal's badness is (1 - confidence), which tops out at 1.0 as
# confidence -> 0; if MISS_WEIGHT were also 1.0, a miss would only barely
# outrank an extremely-low-confidence detection, as an accident of sharing
# the same ceiling, not because missing an animal entirely is actually worse
# than being merely unsure about it. Setting MISS_WEIGHT > 1.0 makes that
# an explicit, tunable statement instead: a miss always outranks any
# confidence-based badness, by a clear margin.


def _visibility_weighted_priority(pool, candidateIdx, rows, matchedConf, missed, maxAnimals, nClasses):
  """For each candidate pool image (candidateIdx, with matching row into
  matchedConf/missed), the single highest-priority animal in it drives the
  image's priority: priority = (mean VIS_WEIGHTS over that animal's
  keypoints) * (MISS_WEIGHT if missed else 1 - confidence). An animal that's
  mostly unlabeled/occluded can't drive selection just because it's also
  low-confidence or missed; an animal that's mostly visible and still missed
  or low-confidence dominates. Returns, per candidate (same order as
  candidateIdx): priority, the driving animal's slot index (-1 if every
  animal in the image was fully unlabeled/padding), its badness, and its
  visibility weight (the latter two for diagnostics only)."""
  annsByImage = {}
  for a in pool['annotations']:
    annsByImage.setdefault(a['image_id'], []).append(a)

  n = len(candidateIdx)
  priority = np.zeros(n)
  drivingSlot = np.full(n, -1)
  drivingBadness = np.full(n, np.nan)
  drivingVis = np.full(n, np.nan)

  for k, (i, row) in enumerate(zip(candidateIdx, rows)):
    imageId = pool['images'][i]['id']
    valid = [a for a in annsByImage.get(imageId, []) if a['num_keypoints'] > 0 and a['area'] > 1]
    for slot, ann in enumerate(valid[:maxAnimals]):
      if missed[row, slot]:
        badness = MISS_WEIGHT
      elif np.isnan(matchedConf[row, slot]):
        continue   # padding slot, not a real animal here
      else:
        badness = 1.0 - matchedConf[row, slot]
      vcodes = np.array(ann['keypoints']).reshape(nClasses, 3)[:, 2]
      visWeight = float(np.mean([VIS_WEIGHTS[int(v)] for v in vcodes]))
      p = visWeight * badness
      if p > priority[k]:
        priority[k], drivingSlot[k], drivingBadness[k], drivingVis[k] = p, slot, badness, visWeight

  return priority, drivingSlot, drivingBadness, drivingVis


SUPPRESS_RADIUS = 100
# Nearby labeled frames (ChimpACT labels every 10th video frame) show the
# same animal in nearly the same pose/occlusion, so they get nearly the same
# priority -- pure top-n_add selection was observed to scoop up whole
# unbroken runs of a clip's labeled frames at once (round 3 confidence: 13/13
# selections from one clip were consecutive, 18 in a row from another) rather
# than spreading the round's budget across more distinct situations.
# SUPPRESS_RADIUS is in raw video-frame units (matching pool images' 'frm'),
# i.e. ~10 labeled-frame steps around each chosen frame get skipped for the
# rest of this round's selection.


def _select_with_suppression(candidateIdx, priority, pool, n_add, suppressRadius=SUPPRESS_RADIUS):
  """Greedily pick the n_add highest-priority candidates, but after picking
  one, skip other candidates from the same clip within suppressRadius raw
  frames of it for the rest of this pass -- so one hard frame "claims" its
  temporal neighborhood instead of also taking its near-duplicates. If
  suppression leaves fewer than n_add reachable (radius too large relative
  to the round's budget), fills the remainder from the highest-priority
  leftovers regardless of suppression, so the round's schedule size is
  always met exactly."""
  order = np.argsort(-priority)
  clipOf = [pool['images'][i]['file_name'].split('/')[0] for i in candidateIdx]
  frmOf = [pool['images'][i]['frm'] for i in candidateIdx]

  chosen, chosenFrmByClip = [], defaultdict(list)
  for k in order:
    if len(chosen) >= n_add:
      break
    clip, frm = clipOf[k], frmOf[k]
    if any(abs(frm - f) <= suppressRadius for f in chosenFrmByClip[clip]):
      continue
    chosen.append(k)
    chosenFrmByClip[clip].append(frm)

  if len(chosen) < n_add:
    chosenSet = set(chosen)
    for k in order:
      if len(chosen) >= n_add:
        break
      if k not in chosenSet:
        chosen.append(k)

  return np.array(candidateIdx)[chosen]


def _build_round_confidence(ds, round_, strategy, n_add):
  """Build round `round_`'s train_TF.json by adding the n_add lowest-
  confidence pool images (missed detections first), per the previous
  round's checkpoint. Needs a GPU + the APT container. Idempotent. Only
  ever called with the base strategy ('confidence') -- 'confidence_warm'
  reuses this round's data verbatim instead (see _copy_round_from_base)."""
  cacheDir = round_dir(ds, round_, strategy)
  jsonPath = os.path.join(cacheDir, 'train_TF.json')
  if os.path.exists(jsonPath):
    return jsonPath

  import PoseTools as pt

  prevDir = round_dir(ds, round_ - 1, strategy if round_ > 1 else None)
  prevData = load_json(os.path.join(prevDir, 'train_TF.json'))
  usedIds = set(im['id'] for im in prevData['images'])

  pool = load_json(ds.pool_json)
  candidateIdx = [i for i, im in enumerate(pool['images']) if im['id'] not in usedIds]
  assert len(candidateIdx) >= n_add, 'pool exhausted'

  prevTrainCacheDir = train_cache_dir(ds, round_ - 1, strategy if round_ > 1 else None)
  confData = pt.pickle_load(os.path.join(prevTrainCacheDir, 'traindata'))[1]
  confData.cachedir = prevTrainCacheDir
  confData.coco_im_dir = ds.coco_im_dir
  matchedConf, missed, info = _pool_confidence(ds, confData, ds.pool_json)
  poolInfo = [(im.get('movid'), im.get('frm')) for im in pool['images']]
  infoToRow = {tuple(x): r for r, x in enumerate(info.tolist())}
  rows = np.array([infoToRow[poolInfo[i]] for i in candidateIdx])

  priority, drivingSlot, drivingBadness, drivingVis = _visibility_weighted_priority(
      pool, candidateIdx, rows, matchedConf, missed, confData.max_n_animals, confData.n_classes)
  chosen = _select_with_suppression(candidateIdx, priority, pool, n_add)

  chosenSet = set(chosen.tolist())
  _write_selection_diagnostics(cacheDir, pool, candidateIdx, priority, drivingSlot, drivingBadness,
                                drivingVis, chosenSet)
  _write_animal_diagnostics(cacheDir, pool, candidateIdx, rows, matchedConf, missed,
                             confData.max_n_animals, confData.n_classes, chosenSet)

  chosenIds = set(pool['images'][i]['id'] for i in chosen)
  newRound = copy.deepcopy(prevData)
  newRound['images'] += [pool['images'][i] for i in chosen]
  newRound['annotations'] += [a for a in pool['annotations'] if a['image_id'] in chosenIds]
  _write_round_json(jsonPath, train_cache_dir(ds, round_, strategy), newRound)
  print(f'{ds.name} round {round_} ({strategy}): added {len(chosen)}, '
        f'total {len(newRound["images"])} images / {len(newRound["annotations"])} annotations')
  return jsonPath


def _write_selection_diagnostics(cacheDir, pool, candidateIdx, priority, drivingSlot, drivingBadness,
                                  drivingVis, chosenSet):
  """Write cacheDir/candidate_confidence.csv -- every candidate pool image
  this round considered, its visibility-weighted priority (see
  _visibility_weighted_priority), the driving animal's slot index (its
  position among that image's non-padding annotations, so a later reader
  can map back to the exact GT annotation -- e.g. for bbox size -- without
  recomputing anything), badness (1.0 if missed else 1-confidence) and
  visibility weight, and whether it got selected -- and print a summary.
  Load the csv (plain text, no APT imports needed) to check the selection:
  are chosen images actually driven by well-visible-but-hard animals, or is
  the pool mostly occluded animals with nothing better to pick?"""
  import csv
  annsByImage = {}
  for a in pool['annotations']:
    annsByImage.setdefault(a['image_id'], []).append(a)

  path = os.path.join(cacheDir, 'candidate_confidence.csv')
  with open(path, 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['image_id', 'movid', 'frm', 'file_name', 'priority', 'driving_slot', 'driving_badness',
                'driving_vis_weight', 'driving_bbox_area', 'selected'])
    for k, i in enumerate(candidateIdx):
      im = pool['images'][i]
      slot = int(drivingSlot[k])
      bboxArea = ''
      if slot >= 0:
        valid = [a for a in annsByImage.get(im['id'], []) if a['num_keypoints'] > 0 and a['area'] > 1]
        if slot < len(valid):
          bboxArea = f'{valid[slot]["area"]:.1f}'
      w.writerow([im['id'], im.get('movid'), im.get('frm'), im.get('file_name', ''),
                  f'{priority[k]:.4f}', slot,
                  '' if np.isnan(drivingBadness[k]) else f'{drivingBadness[k]:.4f}',
                  '' if np.isnan(drivingVis[k]) else f'{drivingVis[k]:.3f}',
                  bboxArea,
                  i in chosenSet])

  chosenSetArr = np.array([i in chosenSet for i in candidateIdx])
  chosenMissedByVis = drivingBadness[chosenSetArr] >= MISS_WEIGHT
  chosenPriority = priority[chosenSetArr]

  print(f'  selection diagnostics -> {path}')
  msg = (f'  chosen: {chosenSetArr.sum()} ({int(chosenMissedByVis.sum())} driven by a missed animal, '
         f'{int((~chosenMissedByVis).sum())} by low confidence')
  if chosenPriority.size:
    msg += f', priority range [{chosenPriority.min():.3f}, {chosenPriority.max():.3f}]'
  print(msg + ')')
  if priority.size:
    print(f'  all {len(candidateIdx)} candidates: priority median={np.median(priority):.3f}, '
          f'range [{priority.min():.3f}, {priority.max():.3f}]')


def _write_animal_diagnostics(cacheDir, pool, candidateIdx, rows, matchedConf, missed, maxAnimals,
                               nClasses, chosenSet):
  """Write cacheDir/candidate_animals.csv -- one row per (candidate image,
  non-padding animal in it), ranked within the image by confidence (rank 1 =
  lowest-confidence/missed first), with that animal's own badness,
  confidence, visibility weight and bbox area. Unlike
  _write_selection_diagnostics (one row per image, the single
  highest-*priority* animal only, where priority also folds in visibility),
  this ranks purely by confidence/badness, so a reader can show "the 2nd,
  3rd, ... worst animal in this frame" -- what the timeline's per-rank rows
  need -- without recomputing anything. matchedConf/missed are the same
  arrays _visibility_weighted_priority uses, already computed; this is a
  free byproduct, no extra inference."""
  import csv
  annsByImage = {}
  for a in pool['annotations']:
    annsByImage.setdefault(a['image_id'], []).append(a)

  path = os.path.join(cacheDir, 'candidate_animals.csv')
  with open(path, 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['image_id', 'movid', 'frm', 'file_name', 'rank', 'slot', 'badness', 'confidence',
                'vis_weight', 'bbox_area', 'missed', 'selected'])
    for k, i in zip(rows, candidateIdx):
      im = pool['images'][i]
      valid = [a for a in annsByImage.get(im['id'], []) if a['num_keypoints'] > 0 and a['area'] > 1]
      animalRows = []
      for slot, ann in enumerate(valid[:maxAnimals]):
        if missed[k, slot]:
          badness, conf = MISS_WEIGHT, 0.0
        elif np.isnan(matchedConf[k, slot]):
          continue
        else:
          conf = float(matchedConf[k, slot])
          badness = 1.0 - conf
        vcodes = np.array(ann['keypoints']).reshape(nClasses, 3)[:, 2]
        visWeight = float(np.mean([VIS_WEIGHTS[int(v)] for v in vcodes]))
        animalRows.append((slot, badness, conf, visWeight, ann['area'], bool(missed[k, slot])))
      animalRows.sort(key=lambda t: -t[1])
      selected = i in chosenSet
      for rank, (slot, badness, conf, visWeight, bboxArea, wasMissed) in enumerate(animalRows, start=1):
        w.writerow([im['id'], im.get('movid'), im.get('frm'), im.get('file_name', ''), rank, slot,
                    f'{badness:.4f}', f'{conf:.4f}', f'{visWeight:.3f}', f'{bboxArea:.1f}',
                    wasMissed, selected])
  print(f'  per-animal diagnostics -> {path}')


def build_round(ds, round_, strategy, n_add, seed=None):
  """Build round `round_`'s train_TF.json. strategy is one of STRATEGIES.
  A '_warm' strategy reuses its base strategy's data verbatim (n_add/seed
  ignored -- see _copy_round_from_base); otherwise selects n_add new images,
  confidence-ranked (needs the container) or random (doesn't). Idempotent."""
  assert strategy in STRATEGIES, f'strategy must be one of {STRATEGIES}'
  if strategy_is_warm(strategy):
    return _copy_round_from_base(ds, round_, strategy)
  if strategy_base(strategy) == 'random':
    return _build_round_random(ds, round_, strategy, n_add=n_add, seed=seed)
  return _build_round_confidence(ds, round_, strategy, n_add=n_add)


def _coco_keypoint_map(evalJsonPath, conf, predLocs, predConf, info, outPrefix):
  """COCO keypoint mAP (OKS-based): AP, AP50, AP75, AP_medium, AP_large, AR,
  AR50, AR75, AR_medium, AR_large -- the same 10 stats and the same
  COCOeval call as APT_interface.compute_coco_map_list, which produced the
  FRCNN/DETR/Grone-HRFormer pose-comparison numbers (see
  chimpact-gt-pose-comparison memory). Not reused directly: that function's
  ground truth comes from a Matlab list file's embedded 1-based 'labels',
  built for a different calling convention. Here evalJsonPath is already a
  valid COCO json (ds.eval_json) -- used directly as coco_gt, no GT
  conversion needed -- so this only has to build the detections side, from
  classify_db_all's predLocs/predConf, the same way compute_coco_map_list
  does. info entries are (movid, frm) pairs (classify_db_all's own image
  ordering), matched back to evalJsonPath's image ids via its images list.
  """
  evalData = load_json(evalJsonPath)
  movfrmToImageId = {(im.get('movid'), im.get('frm')): im['id'] for im in evalData['images']}
  categoryId = evalData['categories'][0]['id']
  npts = conf.n_classes

  locs, scores = predLocs, predConf
  if locs.ndim == 3:
    locs = locs[:, None]
    scores = scores[:, None] if scores is not None else None

  dt_anns = []
  for ndx, curi in enumerate(info):
    imageId = movfrmToImageId.get((curi[0], curi[1]))
    if imageId is None:
      continue
    for ani in range(locs.shape[1]):
      pl = np.array(locs[ndx, ani], dtype=float)
      isgood = np.all(np.isfinite(pl), axis=-1)
      if not np.any(isgood):
        continue
      if scores is not None:
        ps = np.array(scores[ndx, ani], dtype=float)
        ps[~np.isfinite(ps)] = 0.
      else:
        ps = np.ones(npts)
      pl[~isgood] = 0.
      ps[~isgood] = 0.
      kp = np.concatenate([pl, ps[:, None]], axis=1)
      dt_anns.append({'image_id': int(imageId), 'category_id': int(categoryId),
                       'keypoints': kp.flatten().tolist(), 'score': float(np.mean(ps[isgood]))})

  dt_file = outPrefix + '_coco_dt.json'
  write_json(dt_anns, dt_file)

  try:
    from xtcocotools.coco import COCO
    from xtcocotools.cocoeval import COCOeval
  except ImportError:
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

  coco_gt = COCO(evalJsonPath)
  coco_dt = coco_gt.loadRes(dt_file)
  sigmas = np.full(npts, conf.get('coco_oks_sigma', 0.05))
  try:
    coco_eval = COCOeval(coco_gt, coco_dt, 'keypoints', sigmas=sigmas)
  except TypeError:
    # pycocotools takes the sigmas as a param instead of a constructor arg
    coco_eval = COCOeval(coco_gt, coco_dt, 'keypoints')
    coco_eval.params.kpt_oks_sigmas = sigmas
  coco_eval.evaluate()
  coco_eval.accumulate()
  coco_eval.summarize()

  statNames = ['AP', 'AP50', 'AP75', 'AP_medium', 'AP_large',
               'AR', 'AR50', 'AR75', 'AR_medium', 'AR_large']
  return {name: float(val) for name, val in zip(statNames, coco_eval.stats)}


def cmd_eval(ds, round_, strategy=None, model_file=None, eval_json=None):
  """Mean per-keypoint pixel error AND COCO keypoint mAP (see
  _coco_keypoint_map) of a round's (latest, unless model_file given)
  checkpoint on eval_json (default ds.eval_json). Needs the container.
  Returns a dict: {'mean_err_px', 'AP', 'AP50', 'AP75', 'AP_medium',
  'AP_large', 'AR', 'AR50', 'AR75', 'AR_medium', 'AR_large'}.

  Caches the raw predictions (not just the mean) to <round cache
  dir>/eval_predictions.p -- classify_db_all is the expensive part (~2
  min/round here), so a rerun (a different matching threshold, a
  per-keypoint breakdown, looking at specific failure images) doesn't have
  to pay for inference again. Only cached when model_file is None AND
  eval_json is None (the round's own default result) -- an ad hoc checkpoint
  or a diagnostic eval against a different json (e.g. the round's own
  train_db.json, to check for a coordinate-space bug by seeing whether error
  is still large on images the model was directly trained on) isn't the
  round's default result, not worth caching under the round's name."""
  import APT_interface as apt
  import PoseTools as pt
  import pickle

  cacheDir = train_cache_dir(ds, round_, strategy)
  ckpt = model_file or latest_checkpoint(cacheDir, ds.train_name)
  assert ckpt is not None, f'no checkpoints in {cacheDir} yet'
  evalJson = eval_json or ds.eval_json
  cacheable = model_file is None and eval_json is None

  conf = pt.pickle_load(os.path.join(cacheDir, 'traindata'))[1]
  conf.cachedir = cacheDir
  conf.coco_im_dir = ds.coco_im_dir

  predPath = os.path.join(cacheDir, 'eval_predictions.p')
  if cacheable and os.path.exists(predPath):
    cached = pt.pickle_load(predPath)
    ddMatched = cached['ddMatched']
    predLocs, predConf, info = cached['predLocs'], cached['predConf'], cached['info']
  else:
    # batch_size=1 for inference: classify_db2's partial-last-batch handling
    # leaves stale data in unused slots of its pre-allocated buffer
    # (APT_interface.py flags this itself: "for the last batch will be
    # cruft"); batch_size=1 takes the cleaner full-replacement code path
    # instead (conf.batch_size == 1 branch) and sidesteps it entirely.
    conf.batch_size = 1
    res = apt.classify_db_all(ds.net_type, conf, evalJson, model_file=ckpt, name=ds.train_name)
    predLocs, predConf = res[0]['locs'], res[0].get('conf')   # res[0] is a dict ({'locs', 'conf', ...}),
    gtLocs, info = res[1], res[2]                              # not a plain array -- see _pool_confidence
    dd = np.linalg.norm(predLocs[:, :, None] - gtLocs[:, None], axis=-1)
    ddMatched = find_dist_match(dd)
    if cacheable:
      with open(predPath, 'wb') as f:
        pickle.dump({'predLocs': predLocs, 'predConf': predConf, 'gtLocs': gtLocs,
                     'ddMatched': ddMatched, 'info': info, 'ckpt': ckpt, 'eval_json': evalJson}, f)
      print(f'  saved predictions -> {predPath}')

  meanErr = np.nanmean(ddMatched)
  mapStats = _coco_keypoint_map(evalJson, conf, predLocs, predConf, info,
                                 outPrefix=os.path.join(cacheDir, 'eval'))
  print(f'{ds.name} round {round_} ({strategy}), ckpt {os.path.basename(ckpt)}, eval {os.path.basename(evalJson)}: '
        f'mean matched keypoint error = {meanErr:.2f}px over '
        f'{np.count_nonzero(~np.isnan(ddMatched[..., 0]))} matched animals, '
        f'AP={mapStats["AP"]:.3f} AP50={mapStats["AP50"]:.3f} AR={mapStats["AR"]:.3f}')
  return {'mean_err_px': float(meanErr), **mapStats}


MAP_STAT_NAMES = ['AP', 'AP50', 'AP75', 'AP_medium', 'AP_large',
                  'AR', 'AR50', 'AR75', 'AR_medium', 'AR_large']


def cmd_eval_all(ds, strategies, n_rounds, out_dir=None):
  """Evaluate every round 0..n_rounds of every strategy in `strategies`
  against ds.eval_json, write eval_all.csv (strategy, round, n_images,
  mean_err_px, plus the 10 COCO keypoint mAP stats -- see
  _coco_keypoint_map) and an eval_all.png learning-curve plot (mean error
  and AP vs. training set size, one line per strategy, two panels) to
  out_dir (default ds.base_dir). Skips any round that hasn't finished
  training yet rather than failing outright, so this is safe to rerun
  partway through a `loop`. Round 0 is shared across strategies -- evaluated
  once, reused for every strategy's curve. Needs the container
  (classify_db_all) -- run via a submitted job, not interactively on a
  workstation with no GPU."""
  out_dir = out_dir or ds.base_dir
  os.makedirs(out_dir, exist_ok=True)

  rows = []
  round0Stats = None
  round0N = None
  for strategy in strategies:
    for round_ in range(0, n_rounds + 1):
      thisStrategy = strategy if round_ > 0 else None
      trainCacheDir = train_cache_dir(ds, round_, thisStrategy)
      jsonPath = os.path.join(round_dir(ds, round_, thisStrategy), 'train_TF.json')
      if not (os.path.exists(os.path.join(trainCacheDir, 'traindata')) and os.path.exists(jsonPath)
              and latest_checkpoint(trainCacheDir, ds.train_name) is not None):
        print(f'skipping {strategy} round {round_}: not finished training yet ({trainCacheDir})')
        continue

      nImages = len(load_json(jsonPath)['images'])
      if round_ == 0:
        if round0Stats is None:
          round0Stats = cmd_eval(ds, 0, strategy=None)
          round0N = nImages
        stats = round0Stats
        nImages = round0N
      else:
        stats = cmd_eval(ds, round_, strategy=strategy)
      rows.append({'strategy': strategy, 'round': round_, 'n_images': nImages, **stats})

  import csv
  csvPath = os.path.join(out_dir, 'eval_all.csv')
  with open(csvPath, 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=['strategy', 'round', 'n_images', 'mean_err_px'] + MAP_STAT_NAMES)
    w.writeheader()
    w.writerows(rows)
  print(f'wrote {csvPath}')

  import matplotlib
  matplotlib.use('Agg')
  import matplotlib.pyplot as plt
  fig, (axErr, axAP) = plt.subplots(1, 2, figsize=(13, 5))
  for strategy in strategies:
    sub = sorted((r for r in rows if r['strategy'] == strategy), key=lambda r: r['round'])
    if sub:
      xs = [r['n_images'] for r in sub]
      axErr.plot(xs, [r['mean_err_px'] for r in sub], 'o-', label=strategy)
      axAP.plot(xs, [r['AP'] for r in sub], 'o-', label=strategy)
  axErr.set_xlabel('training set size (images)')
  axErr.set_ylabel('mean matched keypoint error (px) on val set')
  axErr.set_xscale('log')
  axErr.legend()
  axAP.set_xlabel('training set size (images)')
  axAP.set_ylabel('COCO keypoint AP (OKS) on val set')
  axAP.set_xscale('log')
  axAP.legend()
  fig.suptitle(f'{ds.name}: active learning comparison')
  fig.tight_layout()
  pngPath = os.path.join(out_dir, 'eval_all.png')
  fig.savefig(pngPath)
  plt.close(fig)
  print(f'wrote {pngPath}')
  return rows


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
  parser = argparse.ArgumentParser(description=__doc__,
                                    formatter_class=argparse.RawDescriptionHelpFormatter)
  parser.add_argument('dataset', choices=sorted(DATASETS))
  sub = parser.add_subparsers(dest='cmd', required=True)

  pInit = sub.add_parser('init', help='build round 0 (random size0 images) -- bare host or container')
  pInit.add_argument('--size0', type=int, default=100)
  pInit.add_argument('--seed', type=int, default=0)

  pRound = sub.add_parser('round', help="build one round's selection -- confidence needs the container")
  pRound.add_argument('--round', type=int, required=True)
  pRound.add_argument('--strategy', choices=STRATEGIES, required=True)
  pRound.add_argument('--n_add', type=int, default=None,
                       help='images to add this round; if omitted, derived from the log-spaced '
                            'schedule via --n_rounds/--size0/--frac_final (same schedule `loop` uses)')
  pRound.add_argument('--n_rounds', type=int, default=8, help='total rounds in the schedule, for --n_add derivation')
  pRound.add_argument('--size0', type=int, default=100)
  pRound.add_argument('--frac_final', type=float, default=1.0)
  pRound.add_argument('--seed', type=int, default=None)

  pLoop = sub.add_parser('loop', help='drive rounds 0..n_rounds unattended -- bare host only')
  pLoop.add_argument('--strategy', choices=STRATEGIES, required=True)
  pLoop.add_argument('--n_rounds', type=int, default=8)
  pLoop.add_argument('--dl_steps', type=int, default=40000)   # calibrated via the chimpact Step-1/Step-2 run
  pLoop.add_argument('--save_step', type=int, default=20000)
  pLoop.add_argument('--size0', type=int, default=100)
  pLoop.add_argument('--frac_final', type=float, default=1.0)
  pLoop.add_argument('--seed', type=int, default=0)
  pLoop.add_argument('--poll_interval', type=int, default=300)
  pLoop.add_argument('--queue', default='gpu_h100')
  pLoop.add_argument('--walltime_min', type=int, default=4320)
  pLoop.add_argument('--lr_drop_step', type=float, default=None,
                      help="override PoseCommon_pytorch's lr_drop_step conf param (default 0.15) for "
                           "every round this call submits -- see cmd_loop's docstring. Pass something "
                           "close to 1.0 (e.g. 0.95) for a '_warm' strategy so it trains at a reduced "
                           "LR for nearly the whole round instead of starting its already-converged "
                           "weights back at full LR.")

  pEval = sub.add_parser('eval', help='score a round -- needs the container')
  pEval.add_argument('--round', type=int, required=True)
  pEval.add_argument('--strategy', choices=STRATEGIES, default=None)
  pEval.add_argument('--model_file', default=None)

  pEvalAll = sub.add_parser('eval_all', help='score every round of every strategy, write csv+plot -- needs the container')
  pEvalAll.add_argument('--strategies', nargs='*', choices=STRATEGIES, default=['confidence', 'random'])
  pEvalAll.add_argument('--n_rounds', type=int, default=8)
  pEvalAll.add_argument('--out_dir', default=None)

  args = parser.parse_args()
  ds = DATASETS[args.dataset]

  if args.cmd == 'init':
    build_round0(ds, size0=args.size0, seed=args.seed)
  elif args.cmd == 'round':
    n_add = args.n_add
    if n_add is None:
      poolSize = len(load_json(ds.pool_json)['images'])
      sizes = round_sizes(poolSize, size0=args.size0, n_rounds=args.n_rounds, frac_final=args.frac_final)
      assert 0 < args.round <= args.n_rounds, f'--round must be in 1..{args.n_rounds}'
      n_add = sizes[args.round] - sizes[args.round - 1]
      print(f'derived n_add={n_add} from schedule {sizes}')
    build_round(ds, args.round, args.strategy, n_add=n_add, seed=args.seed)
  elif args.cmd == 'loop':
    cmd_loop(ds, args.strategy, args.n_rounds, dl_steps=args.dl_steps,
              save_step=args.save_step, size0=args.size0, frac_final=args.frac_final, seed=args.seed,
              poll_interval=args.poll_interval, queue=args.queue, walltime_min=args.walltime_min,
              lr_drop_step=args.lr_drop_step)
  elif args.cmd == 'eval':
    cmd_eval(ds, args.round, strategy=args.strategy, model_file=args.model_file)
  elif args.cmd == 'eval_all':
    cmd_eval_all(ds, args.strategies, args.n_rounds, out_dir=args.out_dir)


if __name__ == '__main__':
  main()

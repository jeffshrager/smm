import argparse, os, time, json
from datetime import datetime
import numpy as np

from smm_core import SMM, calculate_confidence
from finger_counting import FingerCounter
from curriculum import GaussianCurriculum
from data_gen import generate_all_counting_problems, generate_all_addition_problems


def setup_logging(outdir):
    os.makedirs(outdir, exist_ok=True)
    ts = datetime.now().strftime('%Y%m%d%H%M%S')
    tsv = os.path.join(outdir, f'{ts}.tsv')
    out = os.path.join(outdir, f'{ts}.out')
    ftsv = open(tsv, 'w')
    ftsv.write('\t'.join([
        'timestamp','phase','step','addend1','operator','addend2','target','predicted',
        'confidence','used_finger_counting','loss','confidence_criterion','learning_rate','finger_phase',
        'softmax_probs'  # RR: New column for softmax probabilities
    ]) + '\n')
    fout = open(out, 'w')
    return ftsv, fout, ts  # ts is run_id


def log_output(fout, msg):
    now = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
    fout.write(f'[{now}] {msg}\n')
    fout.flush()


def log_step(ftsv, smm, *, phase, step, a1, op, a2, target, predicted, confidence,
             used_finger_counting, loss, probs, finger_phase = ""):
    ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
    cc = getattr(smm, 'confidence_criterion', np.nan)
    lr = getattr(smm, 'learning_rate', np.nan)

    # RR: Convert probs to a string representation
    probs_str = json.dumps(probs.tolist())
    row = [
        ts, phase, step,
        ("" if a1 is None else a1),
        op,
        ("" if a2 is None else a2),
        target, predicted,
        f"{confidence:.6f}",
        str(bool(used_finger_counting)),
        f"{loss:.6f}",
        f"{cc:.6f}",
        f"{lr:.6f}",
        finger_phase, # RR: Directly passed instead of getattr from smm class
        probs_str # RR
    ]
    ftsv.write('\t'.join(map(str, row)) + '\n')
    ftsv.flush()


def quick_report(fout, smm):
    log_output(fout, "=== Sanity check ===")
    tests = [
        (1,'->',None),(3,'->',None),(5,'->',None),
        (2,'->',3),(4,'->',5),
        (1,'+',2),(2,'+',2),(3,'+',4),(4,'+',5),(5,'+',5)
    ]
    for a1, op, a2 in tests:
        if op == '+':
            pred, conf, used_fc = smm.predict_with_finger_counting(a1, op, a2)
            expected = a1 + a2
        else:
            pred, conf, used_fc = smm.predict(a1, op, a2)
            expected = (a1 + 1) if a2 is None else (a2 + 1)
        ok = "+" if pred == expected else "-"
        a1s = str(a1) if a1 is not None else "?"
        a2s = str(a2) if a2 is not None else "?"
        log_output(fout, f"{a1s} {op} {a2s} = {pred} (expected {expected}) {ok} "
                         f"conf:{conf:.3f} finger:{used_fc}")
    log_output(fout, "")


def train_loop(cfg):
    # Setup logging
    ftsv, fout, run_id = setup_logging(cfg['outdir'])

    # Dump full effective configuration into the log and a JSON sidecar
    try:
        _cfg_json = json.dumps(cfg, sort_keys=True, indent=2)
        log_output(fout, '=== EFFECTIVE CONFIGURATION ===')
        for line in _cfg_json.splitlines():
            log_output(fout, line)
        _config_dump_path = os.path.join(cfg['outdir'], f"{run_id}_effective_config.json")
        with open(_config_dump_path, 'w') as _cf:
            _cf.write(_cfg_json)
    except Exception as _e:
        log_output(fout, f'[WARN] Failed to dump configuration: {_e}')

    # Curriculum

    """
    Curriculum setup
    ----------------
    The curriculum controls *which kinds of problems* the model sees at different
    stages of training. This uses a GaussianCurriculum, which gradually shifts the
    distribution of sampled problems from easy "counting" (n -> n+1) toward harder
    addition (a + b).

    Key parameters configured here (all come from the JSON config):
    - total_steps: length of training run.
    - addition_start_step: training step at which addition problems begin to appear.
    - counting_fade_rate: how quickly pure counting fades once addition is introduced.
    - tf_rate_counting / tf_rate_addition: rate constants for the "time flow" schedule
      that drives complexity means separately for counting vs. addition.
    - cf_variance / cx_variance: variances for the Gaussian samples of "complexity".
      (cf = curriculum flow mean; cx = sampled complexity).
    - cf_min_complexity / cf_max_complexity: bounds for counting problems’ difficulty.
    - addition_cf_min / addition_cf_max: bounds for addition problems’ difficulty.
    - counting_focus_steps: optional early steps where only counting is emphasized.
    - Optional overrides: cf_sd and cx_sd can be supplied as standard deviations;
      if present they overwrite cf_variance/cx_variance.

    Mechanism:
    - At each training step, the curriculum computes a "time flow" scalar in [0,1]
      that increases smoothly with step.
    - From this, it derives target means for counting and addition complexities.
    - Complexity for the current step is sampled from a Gaussian around the mean.
    - Task weights (counting_w, addition_w) determine whether to serve a counting or
      addition problem. Counting fades out once addition is introduced, but not to zero.
    - The selected problem (details, target, complexity) is then passed to the model.

    Rationale:
    This scheduling prevents the model from being overwhelmed by addition too early,
    ensures it first stabilizes on simple counting, and then gradually expands its
    competence to harder tasks. The Gaussian noise around complexity means that the
    model sees a spread of difficulties rather than a rigid curriculum.
    """

    cur = GaussianCurriculum()
    cur.total_steps = cfg['total_steps']
    cur.addition_start_step = cfg['addition_start_step']
    cur.counting_fade_rate = cfg['counting_fade_rate']

    # New: gaussian schedule parameters from config
    cur.tf_rate_counting = cfg['tf_rate_counting']
    cur.tf_rate_addition = cfg['tf_rate_addition']
    if hasattr(cur, 'tf_rate'):  # backward compat if curriculum still uses single rate
        cur.tf_rate = float(cfg.get('tf_rate', cur.tf_rate_addition))

    cur.cf_variance = cfg['cf_variance']
    cur.cx_variance = cfg['cx_variance']
    # Optional SD-based overrides (if present, override the variances)
    if 'cf_sd' in cfg: cur.cf_variance = float(cfg['cf_sd'])**2
    if 'cx_sd' in cfg: cur.cx_variance = float(cfg['cx_sd'])**2
    cur.cf_min_complexity = cfg['cf_min_complexity']
    cur.cf_max_complexity = cfg['cf_max_complexity']
    cur.addition_cf_min = cfg['addition_cf_min']
    cur.addition_cf_max = cfg['addition_cf_max']
    cur.counting_focus_steps = cfg['counting_focus_steps']

    # Model

    """
    Model setup
    -----------
    Instantiate the Small Math Model (SMM) with the chosen hidden size, learning rate,
    and gate-freeze schedule. At this stage the model has randomly initialized weights
    and embeddings.

    Parameters injected from config:
    - hidden_size: width of the single hidden (ReLU) layer.
    - learning_rate: initial SGD step size for all parameters.
    - gate_freeze_until_step: number of steps to keep the gating matrix/bias frozen
      at their neutral (zero) initialization. This prevents the gating mechanism from
      dominating too early, forcing the embeddings and hidden layer to learn first.

    Other model properties (embed_size, vocab sizes, etc.) use defaults set inside
    smm_core.SMM unless overridden here.

    Confidence criterion:
    - smm.confidence_criterion is initialized from config['confidence_criterion_start'].
      This is the threshold below which the model will fall back to finger counting.
      Over time, the criterion is annealed downward (later in the training loop),
      so that finger counting is phased out once the model becomes more reliable.

    Notes:
    - At this point the model is structurally complete but has no knowledge; all
      competence comes from the iterative calls to learn_single / learn_addition_with_finger_counting
      during the training loop.
    - The finger counting helper itself is attached later, after we define the logging
      callback, so smm.finger_counter is still None here.
    """

    smm = SMM(hidden_size=cfg['hidden_size'],
              learning_rate=cfg['learning_rate'],
              gate_freeze_until_step=cfg['gate_freeze_until_step'])
    smm.confidence_criterion = cfg['confidence_criterion_start']

    # Finger counter logging callback (matches smm_core.learn_single log_fn signature)

    """
    Finger counter logging callback
    -------------------------------
    Define a wrapper function `_log_cb` that matches the signature expected by
    SMM.learn_single(..., log_fn=...). This callback takes the raw arguments from
    a learning step and writes a structured row into the TSV log.

    Arguments unpacked from the model:
    - a1, op, a2: the input operands and operator.
    - target: ground-truth answer.
    - predicted: model’s chosen answer (argmax).
    - probs: full probability distribution over outputs.
    - loss: cross-entropy loss for this step.
    - phase: label for the training phase ("continuous", "finger_counting", etc.).
    - finger_phase: optional subphase within finger counting (setup/add/verify).

    Additional values computed here:
    - conf: confidence score = 1 - (entropy / max_entropy). This normalizes how
      peaked the distribution is, with 1 = perfect certainty, 0 = uniform guess.
    - step: global step counter of the model.
    - used_fc: boolean flag marking if this step involved finger counting
      (either explicitly phase=="finger_counting" or via a nonempty finger_phase).

    The callback then calls `log_step(...)`, which appends a TSV row with all of
    these fields plus current learning rate and confidence threshold.

    Why this exists:
    - It decouples model training from logging I/O, keeping the model clean.
    - It guarantees consistent logging format across normal training steps and
      finger counting phases.
    - It allows `FingerCounter` to call learn_single with a log_fn and still
      produce identical rows in the run log.
    """

    def _log_cb(to_log_dict):
        # --- NEW: silence finger-counting substeps ---
        # Only keep the single "main" row when finger counting is actually used.
        if to_log_dict.get("phase") == "finger_counting" and to_log_dict.get("finger_phase") != "main_addition":
            return
        # --------------------------------------------

        log_keys = ["addend1", "operator", "addend2", 
                    "target", "predicted", "prob_dist",
                    "loss", "phase", "finger_phase"]

        a1, op, a2, target, predicted, probs, loss, phase, finger_phase = [to_log_dict[key] for key in log_keys]

        conf = calculate_confidence(probs, smm.output_size)
        step = getattr(smm, "step", 0)

        # --- CHANGED: mark used_finger_counting only on the single main row ---
        used_fc = (finger_phase == "main_addition")
        # ---------------------------------------------------------------------

        log_step(
            ftsv, smm,
            phase=phase, step=step,
            a1=a1, op=op, a2=a2,
            target=target, predicted=predicted,
            confidence=conf, used_finger_counting=used_fc,
            loss=loss,
            probs=probs, finger_phase=finger_phase
        )

    fc = FingerCounter(smm, log_step_cb=_log_cb)
    smm.finger_counter = fc

    # Problems universe
    counting = generate_all_counting_problems()
    addition  = generate_all_addition_problems()

    # Pre-run info
    log_output(fout, "=== Small Math Model (Content-Gated) ===")
    log_output(fout, f"Network: {smm.input_size} -> {smm.hidden_size} -> {smm.output_size}")
    log_output(fout, f"Training steps: {cfg['total_steps']}   LR: {cfg['learning_rate']}")
    log_output(fout, f"Gate freeze until: {cfg['gate_freeze_until_step']}")
    log_output(fout, f"Confidence start: {cfg['confidence_criterion_start']} (floor {cfg['confidence_floor']})")
    quick_report(fout, smm)

    # Train

    """
    Training loop
    -------------
    Core loop that runs for `total_steps` iterations. At each step:
    
    1. Curriculum sampling:
       - Calls `cur.select_problem(...)` to choose either a counting (n -> n+1)
         or addition (a + b) problem, guided by the Gaussian curriculum schedule.
       - Returns (problem, time_flow, cf_mean, cx, weight). If no problem is valid,
         the loop skips the step.

    2. Learning:
       - For addition ("+"), calls smm.learn_addition_with_finger_counting(...),
         which may fall back to the FingerCounter strategy before updating weights.
       - For counting ("->"), calls smm.learn_single(...) directly.

    3. Annealing:
       - Learning rate decays multiplicatively by lr_decay, floored at
         learning_rate_floor. This prevents the step size from shrinking to zero.
       - Confidence criterion decays multiplicatively by 0.9999, floored at
         confidence_floor. This gradually raises the bar for trusting the model
         alone, phasing out reliance on finger counting.

    4. Logging:
       - Every 1000 steps, writes a summary line to the .out file including:
         step number, curriculum stats (TF, CF, CX, W), loss, learning rate,
         confidence criterion, and elapsed wallclock time.
       - Per-step detailed logs are already written via the log callback.

    Notes
    -----
    - The loop starts from smm.step if resuming; otherwise from 0.
    - Counting and addition problem universes are pre-generated at the top of
      train_loop (see data_gen.py).
    - The loop terminates naturally when `step == total_steps`.
    - After training completes, both the TSV and .out log files are closed,
      and the run_id (timestamp string) is returned to the caller.
    """

    # Uniform counting target sampler
    def _sample_uniform_count_target(output_size: int):
        """Pick target uniformly from {2..output_size} and back-solve a1 = target-1."""
        t = np.random.randint(2, output_size + 1)
        a1 = t - 1
        return a1, t

    start = time.time()
    for step in range(getattr(smm, "step", 0), int(cur.total_steps)):
        prob, t, cf_mean, cx, w = cur.select_problem(step, counting, addition)
        if prob is None:
            continue
        (a1, op, a2), target, _ = prob

        # make counting targets uniform (2..output_size)
        if op == '->':
            a1, target = _sample_uniform_count_target(smm.output_size)
            a2 = None

        if op == '+':
            loss = smm.learn_addition_with_finger_counting(
                a1, a2,
                log_fn=_log_cb,
                phase="continuous"
            )
        else:
            loss = smm.learn_single(
                a1, op, a2, target,
                log_fn=_log_cb,
                phase="continuous"
            )

        # LR & confidence anneal

        """
        LR & confidence annealing
        -------------------------
        Two multiplicative decays applied every step:

        - Learning rate:
            smm.learning_rate = max(floor, smm.learning_rate * lr_decay)
          → Step size shrinks gradually, but never below learning_rate_floor.
          Purpose: stabilize training as the model converges.

        - Confidence criterion:
            smm.confidence_criterion = max(floor, smm.confidence_criterion * 0.9999)
          → Threshold for "use finger counting" shrinks toward the floor.
          Purpose: early on the model defers often to finger counting,
          but over time the bar lowers so it must rely on itself more.

        Together, these schedules reduce overfitting risk (via smaller LR)
        and encourage independence from the fallback strategy (via lower
        confidence threshold).
        """

        smm.learning_rate = max(cfg['learning_rate_floor'], smm.learning_rate * cfg['lr_decay'])
        smm.confidence_criterion = max(cfg['confidence_floor'], smm.confidence_criterion * 0.9999)

        # Periodic log
        if (step + 1) % 1000 == 0:
            elapsed = time.time() - start
            log_output(
                fout,
                (f"Step {step+1:5d}: TF={t:.3f} CF={cf_mean:.1f} CX={cx:.1f} W={w:.2f} "
                 f"Loss={loss:.6f} LR={smm.learning_rate:.6f} CC={smm.confidence_criterion:.3f} "
                 f"({elapsed:.1f}s)")
            )

    log_output(fout, "=== Training complete ===")
    ftsv.close()
    fout.close()
    return run_id


def load_json(path):
    with open(path, 'r') as f:
        return json.load(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    ap.add_argument("--outdir", type=str, default="results")
    ap.add_argument("--override", type=str, nargs="*", default=[],
                    help="Override as key=value (e.g., learning_rate=0.01 tf_rate_addition=0.0002)")
    args = ap.parse_args()

    cfg = load_json(args.config)
    cfg['outdir'] = args.outdir

    # Keep provenance for logging
    cfg['_config_path'] = args.config
    try:
        cfg['_raw_config'] = load_json(args.config)
    except Exception:
        cfg['_raw_config'] = None
    cfg['_overrides'] = args.override

    # defaults for any missing keys
    defaults = dict(
        total_steps=50000,

        # --- Gaussian curriculum knobs (split movement rates) ---
        tf_rate_counting=0.0001,
        tf_rate_addition=0.0001,
        # optional legacy single rate (if curriculum still reads tf_rate)
        tf_rate=0.0001,

        # spreads and ranges
        cf_variance=1.5,     # or set cf_sd in config to override
        cx_variance=0.8,     # or set cx_sd in config to override
        cf_min_complexity=2.0,
        cf_max_complexity=6.0,
        addition_cf_min=2.0,
        addition_cf_max=10.0,
        counting_focus_steps=6000,

        # core training knobs
        learning_rate=0.005,
        learning_rate_floor=0.002,
        lr_decay=0.9999,
        hidden_size=64,
        gate_freeze_until_step=3000,
        addition_start_step=12000,
        counting_fade_rate=0.0001,
        confidence_criterion_start=0.9,
        confidence_floor=0.75,
        checkpoint_interval=10000  # currently unused (no checkpoints)
    )
    for k, v in defaults.items():
        cfg.setdefault(k, v)

    # apply overrides key=value
    for kv in args.override:
        if "=" not in kv:
            continue
        k, v = kv.split("=", 1)
        v = v.strip()
        if v.lower() in ("true", "false"):
            v = (v.lower() == "true")
        else:
            try:
                if "." in v:
                    v = float(v)
                else:
                    v = int(v)
            except ValueError:
                pass
        cfg[k] = v

    run_id = train_loop(cfg)
    print(f"Run complete: {run_id}")


if __name__ == "__main__":
    main()

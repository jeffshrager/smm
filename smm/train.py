import argparse, os, time, json
from datetime import datetime
import numpy as np
from smm_core import SMM, encode_input, calculate_confidence
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
        'confidence','used_finger_counting','loss','confidence_criterion','learning_rate','finger_phase'
    ]) + '\n')
    ftsv.flush()
    fout = open(out, 'w')
    print(f"Logging to: {tsv}")
    print(f"Output logging to: {out}")
    return ftsv, fout, ts

def log_output(fout, msg):
    print(msg)
    fout.write(msg + '\n')
    fout.flush()

def log_step(ftsv, smm, addend1, operator, addend2, target, predicted, probs, loss, phase, finger_phase=""):
    import datetime as _dt
    ts = _dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
    conf = calculate_confidence(probs, smm.output_size)
    used_finger = conf < smm.confidence_criterion
    a1 = '' if addend1 is None else str(addend1)
    a2 = '' if addend2 is None else str(addend2)
    ftsv.write('\t'.join([
        ts, phase, str(smm.step), a1, operator, a2,
        str(target), str(predicted), f'{conf:.4f}', str(used_finger),
        f'{loss:.6f}', f'{smm.confidence_criterion:.3f}', f'{smm.learning_rate:.6f}', finger_phase
    ]) + '\n')
    ftsv.flush()

def save_checkpoint(path, smm:SMM, curriculum:GaussianCurriculum, meta:dict):
    state = smm.get_state()
    np.savez(path, **state, meta=json.dumps(meta))

def load_checkpoint(path, smm:SMM):
    data = np.load(path, allow_pickle=True)
    state = {k: data[k] for k in ['W1','b1','W2','b2','attn_A','attn_b',
                                  'confidence_criterion','learning_rate','step','gate_freeze_until_step']}
    smm.set_state(state)
    meta = json.loads(str(data['meta']))
    return meta

def test_model(smm:SMM, fout, test_cases):
    log_output(fout, "\n--- Testing Model ---")
    for a1, op, a2 in test_cases:
        if op == '+':
            pred, conf, finger = smm.predict_with_finger_counting(a1, op, a2)
            expected = a1 + a2
        else:
            pred, conf, finger = smm.predict(a1, op, a2)
            expected = (a1 + 1) if a2 is None else (a2 + 1)
        ok = "✓" if pred == expected else "✗"
        a1s = str(a1) if a1 is not None else "?"
        a2s = str(a2) if a2 is not None else "?"
        log_output(fout, f"{a1s} {op} {a2s} = {pred} (expected {expected}) {ok} conf:{conf:.3f} finger:{finger}")
    log_output(fout, "")

def train_loop(cfg):
    # Setup logging
    ftsv, fout, run_id = setup_logging(cfg['outdir'])

    # Curriculum
    cur = GaussianCurriculum()
    cur.total_steps = cfg['total_steps']
    cur.addition_start_step = cfg['addition_start_step']
    cur.counting_fade_rate = cfg['counting_fade_rate']

    # Model
    smm = SMM(hidden_size=cfg['hidden_size'],
              learning_rate=cfg['learning_rate'],
              gate_freeze_until_step=cfg['gate_freeze_until_step'])
    smm.confidence_criterion = cfg['confidence_criterion_start']

    # Finger counter
    fc = FingerCounter(smm, log_step_cb=lambda *args, **kw: log_step(ftsv, smm, *args, **kw))
    smm.finger_counter = fc

    # Problems
    counting = generate_all_counting_problems()
    addition = generate_all_addition_problems()

    # Pre-test
    log_output(fout, "=== Small Math Model (Content-Gated) ===")
    log_output(fout, f"Network: {smm.input_size} -> {smm.hidden_size} -> {smm.output_size}")
    log_output(fout, f"Training steps: {cfg['total_steps']}   LR: {cfg['learning_rate']}")
    log_output(fout, f"Gate freeze until: {cfg['gate_freeze_until_step']}   Confidence start: {cfg['confidence_criterion_start']} (floor {cfg['confidence_floor']})")
    test_cases = [
        (1,'->',None), (3,'->',None), (5,'->',None),
        (2,'->',3), (4,'->',5),
        (1,'+',2), (2,'+',2), (3,'+',4), (4,'+',5), (5,'+',5)
    ]
    test_model(smm, fout, test_cases)

    # Resume?
    if cfg.get('resume_from_checkpoint'):
        meta = load_checkpoint(cfg['resume_from_checkpoint'], smm)
        log_output(fout, f"Resumed from {cfg['resume_from_checkpoint']} at step {smm.step}.")

    # Train
    start = time.time()
    for step in range(smm.step, cur.total_steps):
        prob, t, cf_mean, cx, w = cur.select_problem(step, counting, addition)
        if prob is None:
            continue
        (a1, op, a2), target, _ = prob

        if op == '+':
            loss = smm.learn_addition_with_finger_counting(a1, a2,
                       log_fn=lambda *args, **kw: log_step(ftsv, smm, *args, **kw), phase="continuous")
        else:
            loss = smm.learn_single(a1, op, a2, target,
                       log_fn=lambda *args, **kw: log_step(ftsv, smm, *args, **kw), phase="continuous")

        # LR & confidence anneal
        smm.learning_rate = max(cfg['learning_rate_floor'], smm.learning_rate * cfg['lr_decay'])
        smm.confidence_criterion = max(cfg['confidence_floor'], smm.confidence_criterion * cfg['cc_decay']) if 'cc_decay' in cfg else max(cfg['confidence_floor'], smm.confidence_criterion * 0.9999)

        # Periodic log
        if (step+1) % 1000 == 0:
            elapsed = time.time() - start
            log_output(fout, f"Step {step+1:5d}: TF={t:.3f} CF={cf_mean:.1f} CX={cx:.1f} W={w:.2f} "
                              f"Loss={loss:.6f} LR={smm.learning_rate:.6f} CC={smm.confidence_criterion:.3f} ({elapsed:.1f}s)")

        # Checkpointing
        if cfg['checkpoint_interval'] and (step+1) % cfg['checkpoint_interval'] == 0:
            ckpt_path = os.path.join(cfg['outdir'], f"{run_id}_step{step+1}.npz")
            save_checkpoint(ckpt_path, smm, cur, meta={"run_id": run_id})
            log_output(fout, f"[ckpt] saved: {ckpt_path}")

    # Final tests
    log_output(fout, "\n=== FINAL COMPREHENSIVE TESTING ===")
    log_output(fout, "Counting tests:")
    for n in range(1,6):
        pred, conf, _ = smm.predict(n,'->',None)
        exp = n+1
        ok = "✓" if pred == exp else "✗"
        log_output(fout, f"{n} -> ? = {pred} (expected {exp}) {ok} conf:{conf:.3f}")
    log_output(fout, "")

    log_output(fout, "All addition combinations (with finger counting):")
    log_output(fout, "     1    2    3    4    5")
    for a1 in range(1,6):
        row=[]
        for a2 in range(1,6):
            pred, conf, finger = smm.predict_with_finger_counting(a1,'+',a2)
            exp = a1 + a2
            ok = "✓" if pred == exp else "✗"
            mark = "F" if finger else " "
            row.append(f"{pred}{ok}{mark}")
        log_output(fout, f"{a1}: " + " ".join(f"{cell:4s}" for cell in row))

    log_output(fout, "\nLegend: [predicted][✓/✗][F=finger counting used]")
    log_output(fout, "")
    log_output(fout, f"=== FINGER COUNTING STATS ===")
    log_output(fout, f"Total finger sub-steps: {fc.finger_step}")
    log_output(fout, "")

    ftsv.close(); fout.close()
    return run_id

def load_json(path):
    import json
    with open(path,'r') as f:
        return json.load(f)

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    ap.add_argument("--outdir", type=str, default="results")
    ap.add_argument("--override", type=str, nargs="*", default=[],
                    help="Override as key=value (e.g., learning_rate=0.01)")
    args = ap.parse_args()

    cfg = load_json(args.config)
    cfg['outdir'] = args.outdir

    # defaults for any missing keys
    defaults = dict(
        total_steps=50000,
        learning_rate=0.005,
        learning_rate_floor=0.002,
        lr_decay=0.9999,
        hidden_size=64,
        gate_freeze_until_step=3000,
        addition_start_step=12000,
        counting_fade_rate=0.0001,
        confidence_criterion_start=0.9,
        confidence_floor=0.75,
        checkpoint_interval=10000
    )
    for k,v in defaults.items():
        cfg.setdefault(k, v)

    # apply overrides
    for kv in args.override:
        if "=" in kv:
            k, v = kv.split("=",1)
            # basic literal parsing
            if v.lower() in ("true","false"):
                v = (v.lower()=="true")
            else:
                try:
                    if "." in v: v = float(v)
                    else: v = int(v)
                except ValueError:
                    pass
            cfg[k] = v

    run_id = train_loop(cfg)
    print(f"Run complete: {run_id}")

if __name__ == "__main__":
    main()

from typing import Optional
from smm_core import SMM

"""
FingerCounter
-------------
Implements a three-phase finger counting strategy that the SMM can
fall back on when its own confidence is low during addition.

Purpose:
- Provides an external, symbolic reasoning mechanism (like a child
  literally counting on fingers) to generate correct targets for
  addition problems.
- Allows the model to bootstrap: it first imitates finger counting,
  then gradually internalizes addition in its embeddings and weights.

Phases:
1. Setup (phase1_setup):
   Count up from 1 to the first addend (a1), raising fingers.

2. Add (phase2_add):
   Continue counting from a1 up toward (a1 + a2), but capped at 5
   to reflect realistic finger limits. Each step teaches the model
   to predict n -> n+1 transitions.

3. Verify (phase3_verify):
   Recount all fingers from 1 up to the capped total, reinforcing
   the sequence.

Implementation:
- finger_add(a1, a2):
    Orchestrates the three phases and returns the symbolic sum a1+a2.
- _count_to_number(target, phase_name):
    Trains the model step-by-step to count from 1 up to target.
- _count_from_to(start, end, phase_name):
    Trains the model to continue counting from start up to end.

Logging:
- Each call to learn_single() is passed a callback that records the
  step into the TSV log with phase="finger_counting" and a detailed
  finger_phase (setup/add/verify). This makes finger-counting steps
  distinguishable from normal training.

Notes:
- The finger limit (5) is a built-in simplification: the strategy
  cannot literally count beyond one hand.
- This is not just a heuristic; it is integrated into training,
  so the model is directly shaped by the symbolic strategy until
  its own confidence is high enough.
"""

class FingerCounter:
    """Three-phase finger counting for addition (recursive training events)."""
    def __init__(self, smm_model:SMM, log_step_cb):
        self.smm = smm_model
        self.log_step_cb = log_step_cb
        self.finger_step = 0

    """
    finger_add
    ----------
    Orchestrates the full three-phase finger counting procedure for
    addition problems.

    Given addend1 (a1) and addend2 (a2):
    1. Phase 1 ("phase1_setup"): Count up from 1 to a1.
    2. Phase 2 ("phase2_add"): Continue counting from a1 toward a1+a2,
       capped at 5 to simulate a single hand of fingers.
    3. Phase 3 ("phase3_verify"): Recount from 1 up to the capped sum
       to verify consistency.

    Each step trains the model via smm.learn_single(), logged with
    phase="finger_counting" and a detailed finger_phase tag.

    Returns:
    - The symbolic sum a1+a2 (not capped), so the training loop can
      use it as the target label for addition.
    """

    def finger_add(self, addend1:int, addend2:int) -> int:
        target_sum = addend1 + addend2
        max_countable = min(target_sum, 5)

        # Phase 1: count up to a1
        self._count_to_number(min(addend1,5), "phase1_setup")

        # Phase 2: continue from a1 toward sum (capped by 5)
        if addend1 <= 5:
            self._count_from_to(addend1, max_countable, "phase2_add")

        # Phase 3: recount all fingers to verify
        self._count_to_number(max_countable, "phase3_verify")

        return target_sum

    def _count_to_number(self, target:int, phase_name:str):
        current = 1
        while current < target:
            nxt = current + 1
            _pred, _conf, _ = self.smm.predict(current, '->', None)
            self.smm.learn_single(current, '->', None, nxt,
                                  log_fn=lambda *args, **kw: self.log_step_cb(*args, **kw),
                                  phase="finger_counting", finger_phase=phase_name)
            self.finger_step += 1
            current = nxt
        return current

    def _count_from_to(self, start:int, end:int, phase_name:str):
        current = start
        while current < end and current < 5:
            nxt = current + 1
            _pred, _conf, _ = self.smm.predict(current, '->', None)
            self.smm.learn_single(current, '->', None, nxt,
                                  log_fn=lambda *args, **kw: self.log_step_cb(*args, **kw),
                                  phase="finger_counting", finger_phase=phase_name)
            self.finger_step += 1
            current = nxt
        return current

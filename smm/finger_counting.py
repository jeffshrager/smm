from typing import Optional
from smm_core import SMM

class FingerCounter:
    """Three-phase finger counting for addition (recursive training events)."""
    def __init__(self, smm_model:SMM, log_step_cb):
        self.smm = smm_model
        self.log_step_cb = log_step_cb
        self.finger_step = 0

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
                                  log_fn=lambda *args, **kw: self.log_step_cb(*args, **kw, finger_phase=phase_name),
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
                                  log_fn=lambda *args, **kw: self.log_step_cb(*args, **kw, finger_phase=phase_name),
                                  phase="finger_counting", finger_phase=phase_name)
            self.finger_step += 1
            current = nxt
        return current

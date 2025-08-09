import numpy as np
from typing import Optional, Tuple

def softmax(x: np.ndarray) -> np.ndarray:
    if x.ndim == 1:
        x = x.reshape(1, -1)
    x = x - np.max(x, axis=1, keepdims=True)
    e = np.exp(x)
    return e / (np.sum(e, axis=1, keepdims=True) + 1e-12)

def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0, x)

def calculate_confidence(output_probs: np.ndarray, output_size: int) -> float:
    eps = 1e-10
    entropy = -np.sum(output_probs * np.log(output_probs + eps))
    max_entropy = np.log(output_size)
    return 1.0 - (entropy / max_entropy)

def encode_number(num: int, addend_size: int = 5) -> np.ndarray:
    if num < 1 or num > 5:
        raise ValueError(f"Number {num} out of range 1-5")
    v = np.zeros(addend_size)
    v[num - 1] = 1.0
    return v

def encode_operator(op: str, operator_size: int = 2) -> np.ndarray:
    v = np.zeros(operator_size)
    if op == '+':
        v[0] = 1.0
    elif op == '->':
        v[1] = 1.0
    else:
        raise ValueError(f"Unknown operator: {op}")
    return v

def encode_input(addend1: Optional[int], operator: str, addend2: Optional[int]) -> np.ndarray:
    addend_size = 5
    a1 = np.zeros(addend_size) if addend1 is None else encode_number(addend1, addend_size)
    a2 = np.zeros(addend_size) if addend2 is None else encode_number(addend2, addend_size)
    op = encode_operator(operator)
    return np.concatenate([a1, op, a2])

class SMM:
    "
    Small Math Model with domain-agnostic per-dimension gate.
    - Input: 12 dims (5 a1, 2 op, 5 a2)
    - Hidden: configurable
    - Output: 12 classes (1..12)  [addition & next only in this codebase]
    "
    def __init__(self, hidden_size: int = 64, learning_rate: float = 0.005, gate_freeze_until_step:int=3000):
        self.addend_size = 5
        self.operator_size = 2
        self.input_size = self.addend_size*2 + self.operator_size  # 12
        self.output_size = 12
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate

        rng = np.random.RandomState(123)
        self.W1 = rng.randn(self.input_size, self.hidden_size) * 0.1
        self.b1 = np.zeros((1, self.hidden_size))
        self.W2 = rng.randn(self.hidden_size, self.output_size) * 0.1
        self.b2 = np.zeros((1, self.output_size))

        # Domain-agnostic content gate (start neutral)
        self.attn_A = np.zeros((self.input_size, self.input_size))
        self.attn_b = np.zeros((1, self.input_size))

        self.confidence_criterion = 0.9
        self.step = 0
        self.gate_freeze_until_step = gate_freeze_until_step

        self.finger_counter = None  # set externally

    # ---------- forward / predict ----------
    def forward(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if x.ndim == 1:
            x = x.reshape(1, -1)
        s = x @ self.attn_A + self.attn_b
        gate = 1.0 / (1.0 + np.exp(-s))
        attended = x * gate
        z1 = attended @ self.W1 + self.b1
        h = relu(z1)
        z2 = h @ self.W2 + self.b2
        y = softmax(z2)
        return y, h, gate

    def predict(self, a1: Optional[int], op: str, a2: Optional[int]) -> Tuple[int, float, bool]:
        x = encode_input(a1, op, a2)
        probs, _, gate = self.forward(x)
        conf = calculate_confidence(probs[0], self.output_size)
        pred_idx = int(np.argmax(probs[0]))
        pred_val = pred_idx + 1
        use_fingers = conf < self.confidence_criterion
        return pred_val, conf, use_fingers

    def predict_with_finger_counting(self, a1: int, op: str, a2: int, log_file=None, phase="training"):
        pred, conf, use_fingers = self.predict(a1, op, a2)
        if use_fingers and op == '+' and self.finger_counter is not None:
            target = self.finger_counter.finger_add(a1, a2)
            return target, conf, True
        return pred, conf, use_fingers

    # ---------- learning ----------
    def learn_single(self, a1: Optional[int], op: str, a2: Optional[int], target: int,
                     log_fn=None, phase="training", finger_phase=""):
        x = encode_input(a1, op, a2)
        probs, h, gate = self.forward(x)

        y = np.zeros((1, self.output_size))
        if 1 <= target <= 12:
            y[0, target-1] = 1.0

        # Output layer
        dZ2 = probs - y             # (1,12)
        dW2 = h.T @ dZ2             # (H,12)
        db2 = dZ2                   # (1,12)

        # Hidden
        dH = dZ2 @ self.W2.T        # (1,H)
        dZ1 = dH * (h > 0)          # ReLU'
        # Use attended input to compute dW1
        x_row = x.reshape(1, -1)
        s_forward = x_row @ self.attn_A + self.attn_b
        gate_forward = 1.0 / (1.0 + np.exp(-s_forward))
        attended_forward = x_row * gate_forward
        dW1 = attended_forward.T @ dZ1
        db1 = dZ1

        # Gate gradients
        dA_input = dZ1 @ self.W1.T          # upstream into attended
        dgate = dA_input * x_row            # dL/dgate
        s = s_forward
        gate_now = gate_forward
        ds = dgate * gate_now * (1.0 - gate_now)
        dAttnA = x_row.T @ ds
        dAttnB = ds

        # Update
        lr = self.learning_rate
        self.W2 -= lr * dW2
        self.b2 -= lr * db2
        self.W1 -= lr * dW1
        self.b1 -= lr * db1
        if self.step >= self.gate_freeze_until_step:
            self.attn_A -= lr * dAttnA
            self.attn_b -= lr * dAttnB

        loss = -np.sum(y * np.log(probs + 1e-10))

        if log_fn is not None:
            log_fn(a1, op, a2, target, int(np.argmax(probs[0])) + 1, probs[0], loss, phase, finger_phase)

        self.step += 1
        return float(loss)

    def get_state(self):
        return {
            "W1": self.W1, "b1": self.b1, "W2": self.W2, "b2": self.b2,
            "attn_A": self.attn_A, "attn_b": self.attn_b,
            "confidence_criterion": self.confidence_criterion,
            "learning_rate": self.learning_rate,
            "step": self.step,
            "gate_freeze_until_step": self.gate_freeze_until_step
        }

    def set_state(self, state:dict):
        for k,v in state.items():
            setattr(self, k, v)

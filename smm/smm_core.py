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


class SMM:
    """
    Small Math Model with domain-agnostic per-dimension gate.
    Uses learned embeddings for numbers and operators.
    """

    def __init__(
        self,
        hidden_size: int = 64,
        learning_rate: float = 0.005,
        gate_freeze_until_step: int = 3000,
        embed_size: int = 8,
    ):
        # Vocabulary sizes
        self.num_vocab = 12  # numbers 1..12
        self.operator_size = 2  # '+' and '->'
        self.embed_size = embed_size

        # Input sequence: [a1, op, a2]
        self.input_size = self.embed_size * 3
        self.output_size = 12
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate

        rng = np.random.RandomState(123)

        # Embeddings
        self.number_embeddings = rng.randn(self.num_vocab, self.embed_size) * 0.1
        self.operator_embeddings = rng.randn(self.operator_size, self.embed_size) * 0.1

        # Network weights
        self.W1 = rng.randn(self.input_size, self.hidden_size) * 0.1
        self.b1 = np.zeros((1, self.hidden_size))

        self.W_out = rng.randn(self.hidden_size, self.embed_size) * 0.1
        self.output_embeddings = rng.randn(self.output_size, self.embed_size) * 0.1

        # Domain-agnostic content gate (start neutral)
        self.attn_A = np.zeros((self.input_size, self.input_size))
        self.attn_b = np.zeros((1, self.input_size))

        self.confidence_criterion = 0.9
        self.step = 0
        self.gate_freeze_until_step = gate_freeze_until_step

        self.finger_counter = None  # set externally

    # ---------- encoding helpers ----------
    def encode_number(self, num: int) -> np.ndarray:
        if num < 1 or num > self.num_vocab:
            raise ValueError(f"Number {num} out of range 1-{self.num_vocab}")
        return self.number_embeddings[num - 1]

    def encode_operator(self, op: str) -> np.ndarray:
        if op == '+':
            idx = 0
        elif op == '->':
            idx = 1
        else:
            raise ValueError(f"Unknown operator: {op}")
        return self.operator_embeddings[idx]

    def encode_input(
        self, addend1: Optional[int], operator: str, addend2: Optional[int]
    ) -> np.ndarray:
        a1 = (
            np.zeros(self.embed_size)
            if addend1 is None
            else self.encode_number(addend1)
        )
        a2 = (
            np.zeros(self.embed_size)
            if addend2 is None
            else self.encode_number(addend2)
        )
        op = self.encode_operator(operator)
        return np.concatenate([a1, op, a2])

    # ---------- forward / predict ----------
    def forward(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if x.ndim == 1:
            x = x.reshape(1, -1)
        s = x @ self.attn_A + self.attn_b
        gate = 1.0 / (1.0 + np.exp(-s))
        attended = x * gate
        z1 = attended @ self.W1 + self.b1
        h = relu(z1)
        r = h @ self.W_out
        z2 = r @ self.output_embeddings.T
        y = softmax(z2)
        return y, h, gate

    def predict(self, a1: Optional[int], op: str, a2: Optional[int]) -> Tuple[int, float, bool]:
        x = self.encode_input(a1, op, a2)
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
        x = self.encode_input(a1, op, a2)
        probs, h, gate = self.forward(x)

        y = np.zeros((1, self.output_size))
        if 1 <= target <= 12:
            y[0, target-1] = 1.0

        # Output layer
        dZ2 = probs - y                      # (1,12)
        r = h @ self.W_out                   # (1,E)
        dOutputEmb = dZ2.T @ r               # (12,E)
        dR = dZ2 @ self.output_embeddings    # (1,E)
        dW_out = h.T @ dR                    # (H,E)

        # Hidden layer
        dH = dR @ self.W_out.T               # (1,H)
        dZ1 = dH * (h > 0)                   # ReLU'
        x_row = x.reshape(1, -1)
        s_forward = x_row @ self.attn_A + self.attn_b
        gate_forward = 1.0 / (1.0 + np.exp(-s_forward))
        attended_forward = x_row * gate_forward
        dW1 = attended_forward.T @ dZ1
        db1 = dZ1

        # Gate gradients
        dA_input = dZ1 @ self.W1.T
        dgate = dA_input * x_row
        ds = dgate * gate_forward * (1.0 - gate_forward)
        dAttnA = x_row.T @ ds
        dAttnB = ds

        # Gradients for embeddings (through x)
        dx = dA_input * gate_forward
        da1 = dx[:, : self.embed_size]
        dop = dx[:, self.embed_size: 2 * self.embed_size]
        da2 = dx[:, 2 * self.embed_size:]

        # Update
        lr = self.learning_rate
        self.output_embeddings -= lr * dOutputEmb
        self.W_out -= lr * dW_out
        self.W1 -= lr * dW1
        self.b1 -= lr * db1
        if self.step >= self.gate_freeze_until_step:
            self.attn_A -= lr * dAttnA
            self.attn_b -= lr * dAttnB

        if a1 is not None:
            self.number_embeddings[a1 - 1] -= lr * da1.reshape(-1)
        if a2 is not None:
            self.number_embeddings[a2 - 1] -= lr * da2.reshape(-1)
        op_idx = 0 if op == '+' else 1
        self.operator_embeddings[op_idx] -= lr * dop.reshape(-1)

        loss = -np.sum(y * np.log(probs + 1e-10))

        if log_fn is not None:
            log_fn(a1, op, a2, target, int(np.argmax(probs[0])) + 1, probs[0], loss, phase, finger_phase)

        self.step += 1
        return float(loss)

    def get_state(self):
        return {
            "W1": self.W1,
            "b1": self.b1,
            "W_out": self.W_out,
            "output_embeddings": self.output_embeddings,
            "number_embeddings": self.number_embeddings,
            "operator_embeddings": self.operator_embeddings,
            "attn_A": self.attn_A,
            "attn_b": self.attn_b,
            "confidence_criterion": self.confidence_criterion,
            "learning_rate": self.learning_rate,
            "step": self.step,
            "gate_freeze_until_step": self.gate_freeze_until_step,
        }

    def set_state(self, state: dict):
        for k, v in state.items():
            setattr(self, k, v)

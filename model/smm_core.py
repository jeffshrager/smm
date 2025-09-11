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

        """
        Embeddings
        ----------
        Two separate embedding tables are initialized here:

        - number_embeddings: shape (num_vocab, embed_size).
          Each integer token 1..num_vocab is mapped to a learnable vector.
          Example: number 3 → number_embeddings[2].

        - operator_embeddings: shape (operator_size, embed_size).
          Two operators are supported:
            '+'  → operator_embeddings[0]
            '->' → operator_embeddings[1]

        Both tables are initialized with small random Gaussian values (std ≈ 0.1)
        to break symmetry.

        Usage in the model:
        - The helpers encode_number() and encode_operator() return rows from these
          tables.
        - encode_input(a1, op, a2) concatenates the three vectors
          [embed(a1), embed(op), embed(a2)] into a single input of length
          3 * embed_size.
        - If a1 or a2 is None, encode_input inserts a zero vector of length embed_size
          for that slot.

        Training updates:
        - During learn_single(), gradients are propagated back into whichever
          embeddings were actually used in the input.
        - Only the rows corresponding to the specific tokens (numbers/operators)
          from the current example are updated, mimicking how token embeddings
          are trained in language models.

        Rationale:
        Embeddings let the model discover distributed representations of numbers
        and operators, rather than treating them as one-hot or fixed encodings.
        This mirrors how LLMs learn token embeddings and allows the model to
        capture similarities between tokens (e.g., consecutive numbers).
        """

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

        """
        Encoding helpers
        ----------------
        These methods provide a clean interface from symbolic inputs
        (numbers, operators) to their vector embeddings:

        - encode_number(num: int) → np.ndarray
          Returns the embedding for number `num` in [1..num_vocab].
          Raises ValueError if out of range.

        - encode_operator(op: str) → np.ndarray
          Returns the embedding for an operator token.
          Currently supported:
            '+'  → row 0 of operator_embeddings
            '->' → row 1 of operator_embeddings
          Raises ValueError if the operator is unknown.

        - encode_input(a1, op, a2) → np.ndarray
          Concatenates three embeddings into a single input vector of length
          3 * embed_size:
            [embed(a1), embed(op), embed(a2)]
          If a1 or a2 is None, a zero vector of length embed_size is substituted
          for that slot.

        Purpose:
        - Keeps the rest of the model agnostic to raw integers/strings.
        - Provides consistent handling of missing operands (None → zero vector).
        - Mirrors the token-to-embedding pipeline in large language models.

        Notes:
        - These helpers are used by predict(), learn_single(), and other methods
          whenever a symbolic problem (like "2 + 3") needs to be fed into the model.
        """

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

    """
    Forward and prediction
    ----------------------
    These methods convert an encoded input vector into model outputs.

    - forward(x: np.ndarray) → (probs, hidden, gate)
      * x: concatenated embedding vector of shape (3 * embed_size,).
      * Applies the learned gating mechanism:
          s = x @ attn_A + attn_b
          gate = sigmoid(s)
          attended = x * gate
      * Passes attended input through:
          - Linear layer (W1, b1) + ReLU → hidden representation (h).
          - Linear projection (W_out) → intermediate representation (r).
          - Dot with output_embeddings → logits for each possible output.
          - Softmax → probability distribution over outputs.
      * Returns:
          probs: softmax distribution (1 × output_size)
          h: hidden vector (1 × hidden_size)
          gate: per-dimension gate activations (1 × input_size)

    - predict(a1, op, a2) → (pred_val, conf, use_fingers)
      * Encodes inputs via encode_input(), calls forward().
      * Computes confidence using normalized entropy.
      * Selects prediction as argmax(probs).
      * Returns:
          pred_val: predicted integer (1..output_size)
          conf: confidence score in [0,1]
          use_fingers: True if conf < confidence_criterion

    - predict_with_finger_counting(a1, op, a2, ...)
      * Calls predict().
      * If operator is '+' and confidence is below threshold,
        uses the FingerCounter (if attached) to produce a “finger counting”
        result instead of the raw model output.
      * Returns (prediction, confidence, used_finger_flag).

    Notes:
    - The gating mechanism is key: it can emphasize or suppress dimensions of
      the input embeddings, a simplified analogue of attention.
    - Confidence drives the hybrid strategy: low confidence triggers
      external reasoning (finger counting).
    """

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

    """
    Learning: learn_single
    ----------------------
    Perform one supervised learning update for a single problem.

    Workflow:
    1. Encode input (a1, op, a2) → concatenated embeddings.
    2. Forward pass → (probs, hidden, gate).
    3. Construct one-hot target vector y for the correct answer.
    4. Compute gradients manually (NumPy backprop):
       - Output layer:
         dZ2 = probs - y
         Backprop to output_embeddings and W_out.
       - Hidden layer:
         Backprop through ReLU to W1 and b1.
       - Gate:
         Backprop through sigmoid gate to attn_A and attn_b.
       - Embeddings:
         Slice dx to update the specific number/operator embeddings used.
    5. Apply SGD updates with current learning_rate.
       Note: attn_A and attn_b are only updated once step ≥ gate_freeze_until_step.
    6. Increment global step counter.

    Logging:
    - If log_fn is provided, it is called with (a1, op, a2, target,
      predicted, probs, loss, phase, finger_phase). This allows external
      logging without coupling I/O to the model.

    Return:
    - Scalar float(loss), the negative log-likelihood for this example.

    Notes:
    - This is the most complex part of the model: backprop is fully
      implemented by hand in NumPy, unlike in auto-diff frameworks.
    - Shape discipline is critical: all vectors are reshaped to (1, D)
      to avoid accidental broadcasting.
    - Only embeddings corresponding to the actual tokens in this input
      are updated, just as in large language models.
    - The “finger_phase” argument allows finger-counting sub-steps to be
      distinguished in logs, even though the math is identical.
    """

    def learn_single(self, a1: Optional[int], op: str, a2: Optional[int], target: int,
                     log_fn=None, phase="training", finger_phase=""):


        # Encode → Forward
        # Map symbolic tokens to vectors so the network can learn distributed
        # representations of numbers/operators. The forward pass applies an
        # elementwise *gate* over x (a sigmoid over an affine transform of x),
        # then a linear+ReLU hidden, then projects into a class-embedding table
        # before softmax. Returning (probs, h, gate) exposes the internal states
        # used immediately by the manual backprop below.
        #
        # ALGORITHM
        #   x      = [emb(a1) | emb(op) | emb(a2)]           # shape (1, 3E)
        #   s      = x @ attn_A + attn_b                     # (1, 3E)
        #   gate   = σ(s)                                    # (1, 3E)
        #   z1     = (x * gate) @ W1 + b1                    # (1, H)
        #   h      = ReLU(z1)                                # (1, H)
        #   r      = h @ W_out                               # (1, E)
        #   logits = r @ output_embeddings^T                 # (1, 12)
        #   probs  = softmax(logits)                         # (1, 12)

        x = self.encode_input(a1, op, a2)
        probs, h, gate = self.forward(x)

        # One-hot target construction
        # Supervise a 12-class classifier for answers in {1..12}. Create y as a
        # one-hot row vector; ignore out-of-range targets, which yields an all-zero
        # y and thus zero gradient to the output table.
        #
        # SHAPES
        #   y: (1,12)

        y = np.zeros((1, self.output_size))
        if 1 <= target <= 12:
            y[0, target-1] = 1.0

        # Output layer backprop (softmax + cross-entropy)
        # With softmax + negative log-likelihood, ∂L/∂logits = (probs − y).
        # Because the classifier is factorized into (h→r) and (r→class table),
        # we compute gradients for both the output embedding table and W_out.
        #
        # SHAPES & STEPS
        #   dZ2        = probs − y                   # (1,12)
        #   r          = h @ W_out                   # (1,E)  (recomputed for clarity)
        #   dOutputEmb = dZ2^T @ r                   # (12,E) ∂L/∂output_embeddings
        #   dR         = dZ2 @ output_embeddings     # (1,E)
        #   dW_out     = h^T @ dR                    # (H,E)

        dZ2 = probs - y                      # (1,12)
        r = h @ self.W_out                   # (1,E)
        dOutputEmb = dZ2.T @ r               # (12,E)
        dR = dZ2 @ self.output_embeddings    # (1,E)
        dW_out = h.T @ dR                    # (H,E)

        # Hidden layer backprop (ReLU + gated input path)
        # Push gradient from r back to the hidden representation through W_out,
        # apply ReLU′, and compute W1/b1 gradients using the *gated* input that
        # was used in the forward pass (x * gate).
        #
        # SHAPES & STEPS
        #   dH        = dR @ W_out^T                 # (1,H)
        #   dZ1       = dH * (h > 0)                 # (1,H) ReLU derivative (mask)
        #   x_row     = x[None,:]                    # (1,3E)
        #   s_fwd     = x_row @ attn_A + attn_b      # (1,3E)
        #   gate_fwd  = σ(s_fwd)                     # (1,3E)
        #   attended  = x_row * gate_fwd             # (1,3E)
        #   dW1       = attended^T @ dZ1             # (3E,H)
        #   db1       = dZ1                          # (1,H)

        dH = dR @ self.W_out.T               # (1,H)
        dZ1 = dH * (h > 0)                   # ReLU'
        x_row = x.reshape(1, -1)
        s_forward = x_row @ self.attn_A + self.attn_b
        gate_forward = 1.0 / (1.0 + np.exp(-s_forward))
        attended_forward = x_row * gate_forward
        dW1 = attended_forward.T @ dZ1
        db1 = dZ1

        # Gate (attention-style) gradients
        # The gate learns which input features matter for the task, akin to a
        # per-feature attention mask. Since attended = x * gate and
        # gate = σ(x @ attn_A + attn_b), we backprop via the sigmoid chain.
        #
        # SHAPES & STEPS
        #   dA_input  = dZ1 @ W1^T                    # (1,3E) = ∂L/∂(x*gate)
        #   dgate     = dA_input * x_row              # (1,3E)   ∂L/∂gate
        #   ds        = dgate * gate_fwd*(1−gate_fwd) # (1,3E)   ∂L/∂(x@A+b)
        #   dAttnA    = x_row^T @ ds                  # (3E,3E)
        #   dAttnB    = ds                            # (1,3E)
        # NOTE
        #   Updating (attn_A, attn_b) can be fragile early on; the code supports
        #   freezing these until step ≥ gate_freeze_until_step.

        dA_input = dZ1 @ self.W1.T
        dgate = dA_input * x_row
        ds = dgate * gate_forward * (1.0 - gate_forward)
        dAttnA = x_row.T @ ds
        dAttnB = ds

        # Gradients for embeddings via x
        # Because attended = x * gate, ∂L/∂x = dA_input * gate_fwd. The input x is a
        # concatenation of three E-dimensional blocks, so we slice ∂L/∂x back into
        # (da1, dop, da2) to update the correct embedding rows.
        #
        # SHAPES & STEPS
        #   dx   = dA_input * gate_fwd               # (1,3E)
        #   da1  = dx[:, :E]                         # (1,E)
        #   dop  = dx[:, E:2E]                       # (1,E)
        #   da2  = dx[:, 2E:]                        # (1,E)

        dx = dA_input * gate_forward
        da1 = dx[:, : self.embed_size]
        dop = dx[:, self.embed_size: 2 * self.embed_size]
        da2 = dx[:, 2 * self.embed_size:]

        # Parameter updates (+ optional gate freeze)
        # Apply SGD with a scalar learning rate to the dense parameters. To avoid an
        # early “shortcut” where the model learns to null out inputs, the gate can be
        # left frozen for the first N steps and unfrozen later.
        #
        # UPDATED PARAMETERS
        #   output_embeddings, W_out, W1, b1
        #   (attn_A, attn_b) only if step ≥ gate_freeze_until_step

        lr = self.learning_rate
        self.output_embeddings -= lr * dOutputEmb
        self.W_out -= lr * dW_out
        self.W1 -= lr * dW1
        self.b1 -= lr * db1
        if self.step >= self.gate_freeze_until_step:
            self.attn_A -= lr * dAttnA
            self.attn_b -= lr * dAttnB

        # Embedding table row updates (sparse style)
        # Only the rows that were looked up should be updated: the two operand rows
        # (if present) and the single operator row (index 0 for '+', 1 otherwise).
        # This is standard in embedding-based models to keep updates sparse.

        if a1 is not None:
            self.number_embeddings[a1 - 1] -= lr * da1.reshape(-1)
        if a2 is not None:
            self.number_embeddings[a2 - 1] -= lr * da2.reshape(-1)
        op_idx = 0 if op == '+' else 1
        self.operator_embeddings[op_idx] -= lr * dop.reshape(-1)

        # Loss (cross-entropy): 
        # Negative log-likelihood over the predicted distribution. A small epsilon
        # stabilizes the log in the extremely unlikely event of a zero probability.

        loss = -np.sum(y * np.log(probs + 1e-10))

        # Logging Callback:
        # Externalizes side-effects. If provided, log_fn receives the full context:
        #   (a1, op, a2, target, predicted, probs, loss, phase, finger_phase)
        # so that training code can record traces/metrics without cluttering the model.

        if log_fn is not None:
            log_fn(a1, op, a2, target, int(np.argmax(probs[0])) + 1, probs[0], loss, phase, finger_phase)

        # Bookkeeping:
        # Increment the global step counter and return the scalar loss to the caller.

        self.step += 1
        return float(loss)

    """
    Learning: learn_addition_with_finger_counting
    ---------------------------------------------
    Special-case training routine for addition problems (a1 + a2).

    Purpose:
    - Integrates the external FingerCounter strategy into training.
    - When the model is not yet confident on addition, finger counting
      can provide a reliable target. The model then learns from this
      target just like a normal supervised example.

    Workflow:
    1. Call predict_with_finger_counting(a1, "+", a2).
       - If confidence < threshold and a FingerCounter is attached,
         this will return the finger-counted result instead of the
         model’s raw argmax prediction.
    2. Use that result as the “target” for learn_single().
       - This means the model is always trained toward the correct sum,
         but the source of supervision may be finger counting early on.
    3. Pass phase="training" and finger_phase="main_addition" to make
       logs distinguish these events.

    Return:
    - Scalar float(loss) from learn_single().

    Notes:
    - This creates a feedback loop: the symbolic finger-counting
      strategy scaffolds the model until its confidence is high enough
      to stand alone.
    - In logs, these steps appear under phase="training" with
      finger_phase="main_addition", which analysts can filter on.
    - This mechanism mirrors how children may first use external
      counting strategies and later internalize them.
    """

    def learn_addition_with_finger_counting(
        self, addend1: int, addend2: int,
        log_fn=None, phase: str = "training",
    ) -> float:
        """Learn addition using finger counting before training on the result."""
        finger_result, _, _ = self.predict_with_finger_counting(
            addend1, "+", addend2, log_fn, phase
        )
        return self.learn_single(
            addend1, "+", addend2, finger_result,
            log_fn=log_fn, phase=phase, finger_phase="main_addition",
        )

    """
    State serialization
    -------------------
    Methods for saving and restoring model parameters.

    - get_state() -> dict
      Returns a Python dictionary containing all learnable parameters
      and key hyperparameters:
        W1, b1, W_out, output_embeddings,
        number_embeddings, operator_embeddings,
        attn_A, attn_b,
        confidence_criterion, learning_rate,
        step, gate_freeze_until_step.

      This dictionary is JSON-serializable if arrays are converted
      to lists (handled externally), or NumPy can save it as a .npz.

    - set_state(state: dict)
      Restores attributes from a state dictionary.
      Simply loops over key/value pairs and sets them as attributes
      on the SMM instance.

    Notes:
    - These functions provide the same role as PyTorch’s
      state_dict() / load_state_dict(), but with simpler naming.
    - Enables checkpointing, resuming, or reproducing training runs.
    - If you plan to integrate with existing ML tooling, consider
      renaming these to match the more common `state_dict` convention.
    - Because numpy arrays are mutable, deep copies may be needed
      when storing multiple states simultaneously.
    """

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

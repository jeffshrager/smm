import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List, Optional
import random
import os
from datetime import datetime
import time

# ===============================================================================
# GLOBAL LOGGING SYSTEM
# ===============================================================================

def setup_logging():
    """Initialize logging system"""
    # Create results directory if it doesn't exist
    os.makedirs('results', exist_ok=True)
    
    # Generate timestamp for log filename
    timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
    log_filename = f'results/{timestamp}.tsv'
    out_filename = f'results/{timestamp}.out'
    
    # Initialize TSV log file with headers
    log_file = open(log_filename, 'w')
    headers = [
        'timestamp', 'phase', 'step', 'addend1', 'operator', 'addend2',
        'target', 'predicted', 'confidence', 'used_finger_counting', 'loss',
        'confidence_criterion', 'learning_rate'
    ]
    log_file.write('\t'.join(headers) + '\n')
    log_file.flush()
    
    # Initialize output log file
    out_file = open(out_filename, 'w')
    
    print(f"Logging to: {log_filename}")
    print(f"Output logging to: {out_filename}")
    return log_file, out_file

def log_output(out_file, message):
    """Log a message to both console and output file"""
    print(message)
    out_file.write(message + '\n')
    out_file.flush()

def log_training_step(log_file, phase: str, step: int, 
                     addend1: Optional[int], operator: str, addend2: Optional[int],
                     target: int, predicted: int, confidence: float, 
                     used_finger_counting: bool, loss: float, confidence_criterion: float,
                     learning_rate: float):
    """Log a single training step"""
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
    
    # Convert None values to empty strings for logging
    a1_str = str(addend1) if addend1 is not None else ''
    a2_str = str(addend2) if addend2 is not None else ''
    
    log_entry = [
        timestamp, phase, str(step), a1_str, operator, a2_str,
        str(target), str(predicted), f'{confidence:.4f}', 
        str(used_finger_counting), f'{loss:.6f}', f'{confidence_criterion:.3f}',
        f'{learning_rate:.6f}'
    ]
    
    log_file.write('\t'.join(log_entry) + '\n')
    log_file.flush()

# ===============================================================================
# ENCODING FUNCTIONS
# ===============================================================================

def encode_number(num: int, addend_size: int = 5) -> np.ndarray:
    """One-hot encode a number (1-5)"""
    if num < 1 or num > 5:
        raise ValueError(f"Number {num} out of range 1-5")
    encoding = np.zeros(addend_size)
    encoding[num - 1] = 1.0
    return encoding

def encode_operator(op: str, operator_size: int = 2) -> np.ndarray:
    """One-hot encode an operator ('+' or '->')"""
    encoding = np.zeros(operator_size)
    if op == '+':
        encoding[0] = 1.0
    elif op == '->':
        encoding[1] = 1.0
    else:
        raise ValueError(f"Unknown operator: {op}")
    return encoding

def encode_input(addend1: Optional[int], operator: str, addend2: Optional[int]) -> np.ndarray:
    """Encode a complete input sequence"""
    addend_size = 5
    
    # Handle single number inputs (e.g., "3 -> ?")
    if addend1 is None:
        enc1 = np.zeros(addend_size)
    else:
        enc1 = encode_number(addend1, addend_size)
        
    if addend2 is None:
        enc2 = np.zeros(addend_size)
    else:
        enc2 = encode_number(addend2, addend_size)
        
    op_enc = encode_operator(operator)
    
    return np.concatenate([enc1, op_enc, enc2])

# ===============================================================================
# ACTIVATION FUNCTIONS
# ===============================================================================

def softmax(x: np.ndarray) -> np.ndarray:
    """Softmax activation function"""
    if x.ndim == 1:
        x = x.reshape(1, -1)
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

def relu(x: np.ndarray) -> np.ndarray:
    """ReLU activation function"""
    return np.maximum(0, x)

# ===============================================================================
# CONFIDENCE CALCULATION
# ===============================================================================

def calculate_confidence(output_probs: np.ndarray, output_size: int = 12) -> float:
    """Calculate confidence based on entropy of output distribution"""
    # Use entropy as confidence measure (lower entropy = higher confidence)
    epsilon = 1e-10  # Prevent log(0)
    entropy = -np.sum(output_probs * np.log(output_probs + epsilon))
    
    # Normalize entropy to 0-1 range (max entropy for uniform distribution)
    max_entropy = np.log(output_size)
    normalized_entropy = entropy / max_entropy
    
    # Convert to confidence (1 - normalized_entropy)
    confidence = 1.0 - normalized_entropy
    return confidence

# ===============================================================================
# DATA GENERATION FUNCTIONS
# ===============================================================================

def generate_all_counting_problems():
    """Generate all possible counting problems with their complexities"""
    problems = []
    
    # Single number counting: "n -> n+1"
    for n in range(1, 6):  # Input numbers 1-5
        target = n + 1
        details = (n, '->', None)
        problems.append((details, target, target))  # (details, target, complexity)
    
    # Two number counting: "n, n+1 -> n+2"
    for n in range(1, 5):  # Both numbers must be ≤ 5
        if n + 1 <= 5:
            target = n + 2
            details = (n, '->', n + 1)
            problems.append((details, target, target))  # complexity = target
    
    # Extended counting for higher targets (using single input patterns)
    for target in range(7, 13):  # Targets 7-12
        # Use the largest valid input that makes sense
        input_num = min(5, target - 1)
        details = (input_num, '->', None)
        problems.append((details, target, target))  # complexity = target
    
    return problems

def generate_all_addition_problems():
    """Generate all possible addition problems with their complexities"""
    problems = []
    
    for a1 in range(1, 6):  # 1-5
        for a2 in range(1, 6):  # 1-5
            target = a1 + a2
            details = (a1, '+', a2)
            problems.append((details, target, target))  # complexity = target (sum)
    
    return problems

# ===============================================================================
# FINGER COUNTING FALLBACK
# ===============================================================================

def finger_counting(addend1: int, addend2: int) -> int:
    """External mechanical method for getting addition answers"""
    return addend1 + addend2

# ===============================================================================
# SMALL MATH MODEL CLASS
# ===============================================================================

class SMM:
    """Small Math Model - Continuous learning neural network"""
    
    def __init__(self, hidden_size: int = 64, learning_rate: float = 0.01):
        # Network dimensions
        self.addend_size = 5  # Numbers 1-5
        self.operator_size = 2  # '+' and '->'
        self.input_size = self.addend_size * 2 + self.operator_size
        self.output_size = 12  # Results 1-12
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate
        
        # Initialize weights
        self.W1 = np.random.randn(self.input_size, self.hidden_size) * 0.1
        self.b1 = np.zeros((1, self.hidden_size))
        self.W2 = np.random.randn(self.hidden_size, self.output_size) * 0.1
        self.b2 = np.zeros((1, self.output_size))
        
        # Simple attention weights (learned)
        self.attention_W = np.random.randn(self.addend_size * 2, 2) * 0.1
        
        # Confidence criterion for finger counting
        self.confidence_criterion = 0.5
        
        # Training step counter
        self.step = 0
    
    def forward(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Forward pass through the network"""
        if x.ndim == 1:
            x = x.reshape(1, -1)
            
        # Simple attention mechanism
        addend_features = np.concatenate([x[:, :self.addend_size], x[:, -self.addend_size:]], axis=1)
        attention_scores = softmax(addend_features @ self.attention_W)
        
        # Apply attention (simplified for now)
        attended_input = x
        
        # Hidden layer
        z1 = attended_input @ self.W1 + self.b1
        a1 = relu(z1)
        
        # Output layer
        z2 = a1 @ self.W2 + self.b2
        a2 = softmax(z2)
        
        return a2, a1, attention_scores
    
    def predict(self, addend1: Optional[int], operator: str, addend2: Optional[int]) -> Tuple[int, float, bool]:
        """Make a prediction and determine if finger counting is needed"""
        x = encode_input(addend1, operator, addend2)
        output_probs, _, attention = self.forward(x)
        
        confidence = calculate_confidence(output_probs[0], self.output_size)
        predicted_idx = np.argmax(output_probs[0])
        predicted_value = predicted_idx + 1  # Convert back to 1-12 range
        
        use_finger_counting = confidence < self.confidence_criterion
        
        return predicted_value, confidence, use_finger_counting
    
    def learn_single(self, addend1: Optional[int], operator: str, addend2: Optional[int], target: int,
                    log_file=None, phase: str = "training") -> float:
        """Learn from a single example (continuous learning)"""
        # Encode input
        x = encode_input(addend1, operator, addend2)
        
        # Forward pass
        output_probs, hidden_activations, attention_scores = self.forward(x)
        
        # Convert target to one-hot
        y_one_hot = np.zeros((1, self.output_size))
        if 1 <= target <= 12:
            y_one_hot[0, target - 1] = 1.0
        
        # Backward pass (single example)
        # Output layer gradients
        dZ2 = output_probs - y_one_hot
        dW2 = hidden_activations.T @ dZ2
        db2 = dZ2
        
        # Hidden layer gradients
        dA1 = dZ2 @ self.W2.T
        dZ1 = dA1 * (hidden_activations > 0)  # ReLU derivative
        dW1 = x.reshape(-1, 1) @ dZ1
        db1 = dZ1
        
        # Update weights
        self.W2 -= self.learning_rate * dW2
        self.b2 -= self.learning_rate * db2
        self.W1 -= self.learning_rate * dW1
        self.b1 -= self.learning_rate * db1
        
        # Calculate loss
        loss = -np.sum(y_one_hot * np.log(output_probs + 1e-10))
        
        # Log this step
        if log_file:
            predicted = np.argmax(output_probs[0]) + 1
            confidence = calculate_confidence(output_probs[0], self.output_size)
            used_finger_counting = confidence < self.confidence_criterion
            
            log_training_step(
                log_file, phase, self.step, addend1, operator, addend2,
                target, predicted, confidence, used_finger_counting, loss,
                self.confidence_criterion, self.learning_rate
            )
        
        self.step += 1
        return loss

# ===============================================================================
# GAUSSIAN CURRICULUM LEARNING SYSTEM
# ===============================================================================

class GaussianCurriculum:
    """Continuous curriculum learning using hierarchical Gaussians"""
    
    def __init__(self):
        # Time Flow (TF) parameters - controls overall pacing
        self.tf_rate = 0.0001  # How fast time flows per step (much slower for continuous)
        
        # Complexity Flow (CF) parameters - gets mean from TF
        self.cf_variance = 1.5  # How spread out complexity sampling is
        self.cf_min_complexity = 2.0  # Minimum complexity (2 for counting 1->2)
        self.cf_max_complexity = 12.0  # Maximum complexity
        
        # Complexity (CX) parameters - gets mean from CF, samples actual problems
        self.cx_variance = 0.8  # Variation around current complexity focus
        
        # Separate parameters for addition vs counting
        self.addition_cf_min = 2.0  # Minimum addition complexity (1+1=2)
        self.addition_cf_max = 10.0  # Maximum addition complexity (5+5=10)
        
        # Training duration (in steps, not epochs)
        self.total_steps = 10000
        self.counting_focus_steps = 6000  # Steps to focus on counting
        self.addition_start_step = 3000  # When addition begins (overlap)
        
        # Mixing parameters
        self.counting_fade_rate = 0.0002  # How fast counting fades after addition starts
        
    def get_time_flow(self, step):
        """TF: Convert step to curriculum time (0 to 1)"""
        return min(1.0, step * self.tf_rate)
    
    def get_complexity_flow_mean(self, time_flow, task_type="counting"):
        """CF: Get complexity mean based on time flow"""
        if task_type == "counting":
            min_complexity = self.cf_min_complexity
            max_complexity = self.cf_max_complexity
        else:  # addition
            min_complexity = self.addition_cf_min
            max_complexity = self.addition_cf_max
            
        # Linear interpolation from min to max complexity
        return min_complexity + time_flow * (max_complexity - min_complexity)
    
    def sample_complexity(self, cf_mean):
        """CX: Sample actual complexity around the CF mean"""
        complexity = np.random.normal(cf_mean, self.cx_variance)
        # Clamp to valid range
        return max(2, min(12, complexity))
    
    def get_task_weights(self, step):
        """Determine relative weights for counting vs addition"""
        counting_weight = 1.0
        addition_weight = 0.0
        
        if step >= self.addition_start_step:
            # Addition starts and grows
            addition_progress = (step - self.addition_start_step) / (self.total_steps - self.addition_start_step)
            addition_weight = min(1.0, addition_progress * 2)  # Ramp up addition
            
            # Counting fades gradually
            fade_progress = (step - self.addition_start_step) * self.counting_fade_rate
            counting_weight = max(0.0, 1.0 - fade_progress)
        
        return counting_weight, addition_weight
    
    def select_problem(self, step, counting_problems, addition_problems):
        """Select a single problem based on current curriculum state"""
        # Get curriculum state
        time_flow = self.get_time_flow(step)
        counting_cf_mean = self.get_complexity_flow_mean(time_flow, "counting")
        addition_cf_mean = self.get_complexity_flow_mean(time_flow, "addition")
        counting_weight, addition_weight = self.get_task_weights(step)
        
        # Sample complexities
        counting_complexity = self.sample_complexity(counting_cf_mean)
        addition_complexity = self.sample_complexity(addition_cf_mean)
        
        # Choose task type based on weights
        total_weight = counting_weight + addition_weight
        if total_weight <= 0:
            return None, 0, 0, 0, 0  # No valid tasks
            
        task_choice = random.random()
        if task_choice < counting_weight / total_weight:
            # Select counting problem
            valid_problems = [
                p for p in counting_problems 
                if abs(p[2] - counting_complexity) <= 1.0
            ]
            if not valid_problems:
                valid_problems = counting_problems
            
            problem = random.choice(valid_problems)
            return problem, time_flow, counting_cf_mean, counting_complexity, counting_weight
        else:
            # Select addition problem
            valid_problems = [
                p for p in addition_problems 
                if abs(p[2] - addition_complexity) <= 1.0
            ]
            if not valid_problems:
                valid_problems = addition_problems
            
            problem = random.choice(valid_problems)
            return problem, time_flow, addition_cf_mean, addition_complexity, addition_weight
    
    def log_config(self, out_file):
        """Log all curriculum parameters"""
        log_output(out_file, "=== GAUSSIAN CURRICULUM CONFIGURATION (CONTINUOUS) ===")
        log_output(out_file, f"Time Flow (TF):")
        log_output(out_file, f"  TF rate: {self.tf_rate} (curriculum speed per step)")
        log_output(out_file, f"")
        log_output(out_file, f"Complexity Flow (CF):")
        log_output(out_file, f"  CF variance: {self.cf_variance} (complexity spread)")
        log_output(out_file, f"  Counting complexity range: {self.cf_min_complexity} -> {self.cf_max_complexity}")
        log_output(out_file, f"  Addition complexity range: {self.addition_cf_min} -> {self.addition_cf_max}")
        log_output(out_file, f"")
        log_output(out_file, f"Complexity (CX):")
        log_output(out_file, f"  CX variance: {self.cx_variance} (problem variation)")
        log_output(out_file, f"")
        log_output(out_file, f"Training schedule:")
        log_output(out_file, f"  Total steps: {self.total_steps}")
        log_output(out_file, f"  Counting focus steps: {self.counting_focus_steps}")
        log_output(out_file, f"  Addition starts at step: {self.addition_start_step}")
        log_output(out_file, f"  Counting fade rate: {self.counting_fade_rate}")
        log_output(out_file, "")

# ===============================================================================
# CONTINUOUS TRAINING FUNCTION
# ===============================================================================

def train_continuous_curriculum(smm, curriculum, counting_problems, addition_problems, log_file, out_file):
    """Continuous learning with Gaussian curriculum"""
    
    log_output(out_file, f"=== CONTINUOUS CURRICULUM TRAINING ({curriculum.total_steps} steps) ===")
    log_output(out_file, "Learning one problem at a time...")
    log_output(out_file, "")
    
    start_time = time.time()
    
    for step in range(curriculum.total_steps):
        # Select problem based on curriculum
        problem_data = curriculum.select_problem(step, counting_problems, addition_problems)
        if problem_data[0] is None:
            continue
            
        problem, time_flow, cf_mean, complexity, weight = problem_data
        details, target, _ = problem
        addend1, operator, addend2 = details
        
        # Learn from this single problem
        loss = smm.learn_single(addend1, operator, addend2, target, log_file, "continuous")
        
        # Update learning rate and confidence criterion continuously
        lr_decay = 0.9999  # Very gradual decay
        cc_decay = 0.9999
        smm.learning_rate = max(0.001, smm.learning_rate * lr_decay)
        smm.confidence_criterion = max(0.6, smm.confidence_criterion * cc_decay)
        
        # Log progress every 1000 steps
        if (step + 1) % 1000 == 0:
            elapsed = time.time() - start_time
            log_output(out_file, 
                f"Step {step + 1:5d}: TF={time_flow:.3f} "
                f"CF={cf_mean:.1f} CX={complexity:.1f} W={weight:.2f} "
                f"Loss={loss:.6f} LR={smm.learning_rate:.6f} CC={smm.confidence_criterion:.3f} "
                f"({elapsed:.1f}s)")

# ===============================================================================
# TESTING FUNCTIONS
# ===============================================================================

def test_model(smm, out_file, test_cases):
    """Test the model on specific cases"""
    log_output(out_file, "\n--- Testing Model ---")
    for addend1, op, addend2 in test_cases:
        pred, conf, finger = smm.predict(addend1, op, addend2)
        if op == '+':
            expected = addend1 + addend2
        elif addend2 is None:
            expected = addend1 + 1
        else:
            expected = addend2 + 1
        
        correct = "✓" if pred == expected else "✗"
        
        a1_str = str(addend1) if addend1 is not None else "?"
        a2_str = str(addend2) if addend2 is not None else "?"
        
        log_output(out_file, f"{a1_str} {op} {a2_str} = {pred} (expected {expected}) {correct} conf:{conf:.3f} finger:{finger}")
    log_output(out_file, "")

# ===============================================================================
# MAIN EXECUTION
# ===============================================================================

if __name__ == "__main__":
    # Setup logging
    log_file, out_file = setup_logging()
    
    # Initialize Gaussian curriculum
    curriculum = GaussianCurriculum()
    curriculum.log_config(out_file)
    
    # Initialize the model
    smm = SMM(hidden_size=64, learning_rate=0.02)
    smm.confidence_criterion = 0.9
    
    log_output(out_file, "=== Small Math Model (Continuous Learning) ===")
    log_output(out_file, f"Network: {smm.input_size} -> {smm.hidden_size} -> {smm.output_size}")
    log_output(out_file, f"Initial learning rate: {smm.learning_rate}")
    log_output(out_file, f"Initial confidence criterion: {smm.confidence_criterion}")
    log_output(out_file, "")
    
    # Generate all problems with complexities
    counting_problems = generate_all_counting_problems()
    addition_problems = generate_all_addition_problems()
    
    log_output(out_file, "Generated problem sets:")
    log_output(out_file, f"  Counting problems: {len(counting_problems)}")
    log_output(out_file, f"  Addition problems: {len(addition_problems)}")
    log_output(out_file, "")
    
    # Show complexity ranges
    counting_complexities = [c for _, _, c in counting_problems]
    addition_complexities = [c for _, _, c in addition_problems]
    
    log_output(out_file, "Complexity ranges:")
    log_output(out_file, f"  Counting: {min(counting_complexities)} to {max(counting_complexities)}")
    log_output(out_file, f"  Addition: {min(addition_complexities)} to {max(addition_complexities)}")
    log_output(out_file, "")
    
    # Test before training
    log_output(out_file, "=== TESTING BEFORE TRAINING ===")
    test_cases = [
        (1, '->', None), (3, '->', None), (5, '->', None),  # Simple counting
        (2, '->', 3), (4, '->', 5),                         # Sequence counting  
        (1, '+', 2), (2, '+', 2),                           # Small addition
        (3, '+', 4), (4, '+', 5), (5, '+', 5)               # Large addition
    ]
    test_model(smm, out_file, test_cases)
    
    # Continuous training with Gaussian curriculum
    train_continuous_curriculum(smm, curriculum, counting_problems, addition_problems, log_file, out_file)
    
    # Final comprehensive test
    log_output(out_file, "\n=== FINAL COMPREHENSIVE TESTING ===")
    
    # Test counting across complexity range
    log_output(out_file, "Counting tests:")
    for n in range(1, 6):
        pred, conf, finger = smm.predict(n, '->', None)
        expected = n + 1
        correct = "✓" if pred == expected else "✗"
        log_output(out_file, f"{n} -> ? = {pred} (expected {expected}) {correct} conf:{conf:.3f}")
    log_output(out_file, "")
    
    # Test all addition combinations with complexity analysis
    log_output(out_file, "All addition combinations (complexity analysis):")
    log_output(out_file, "     1    2    3    4    5")
    for a1 in range(1, 6):
        row = []
        for a2 in range(1, 6):
            pred, conf, finger = smm.predict(a1, '+', a2)
            expected = a1 + a2
            correct = "✓" if pred == expected else "✗"
            finger_marker = "F" if finger else " "
            row.append(f"{pred}{correct}{finger_marker}")
        log_output(out_file, f"{a1}: " + " ".join(f"{cell:4s}" for cell in row))
    
    log_output(out_file, "\nLegend: [predicted][✓/✗][F=finger counting needed]")
    log_output(out_file, "")
    
    # Analyze performance by complexity
    log_output(out_file, "Performance by complexity:")
    for complexity in sorted(set(addition_complexities)):
        correct_count = 0
        total_count = 0
        finger_count = 0
        
        for a1 in range(1, 6):
            for a2 in range(1, 6):
                if a1 + a2 == complexity:
                    pred, conf, finger = smm.predict(a1, '+', a2)
                    expected = a1 + a2
                    if pred == expected:
                        correct_count += 1
                    if finger:
                        finger_count += 1
                    total_count += 1
        
        if total_count > 0:
            accuracy = correct_count / total_count * 100
            finger_rate = finger_count / total_count * 100
            log_output(out_file, f"  Complexity {complexity:2d}: {correct_count}/{total_count} correct ({accuracy:5.1f}%), {finger_count} finger counting ({finger_rate:5.1f}%)")
    
    log_output(out_file, "")
    
    # Test the counting vs addition conflict
    log_output(out_file, "Counting vs Addition conflict analysis:")
    conflicts = [
        (2, 3),  # 2->3 vs 2+3=5
        (3, 4),  # 3->4 vs 3+4=7  
        (4, 5),  # 4->5 vs 4+5=9
    ]
    
    for a1, expected_next in conflicts:
        # Test counting
        count_pred, count_conf, count_finger = smm.predict(a1, '->', None)
        # Test addition
        add_pred, add_conf, add_finger = smm.predict(a1, '+', a1)
        
        log_output(out_file, f"  {a1} -> ? = {count_pred} (expected {expected_next}) conf:{count_conf:.3f}")
        log_output(out_file, f"  {a1} + {a1} = {add_pred} (expected {a1*2}) conf:{add_conf:.3f}")
        
        # Check if model learned to distinguish
        count_correct = count_pred == expected_next
        add_correct = add_pred == a1 * 2
        conflict_resolved = count_correct and add_correct
        
        log_output(out_file, f"    Conflict resolved: {conflict_resolved} (counting: {count_correct}, addition: {add_correct})")
        log_output(out_file, "")
    
    # Close log files
    log_file.close()
    out_file.close()
    print("Continuous curriculum training complete!")
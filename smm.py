import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List, Optional
import random
import os
from datetime import datetime

# Global logging setup
def setup_logging():
    """Initialize logging system"""
    # Create results directory if it doesn't exist
    os.makedirs('results', exist_ok=True)
    
    # Generate timestamp for log filename
    timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
    log_filename = f'results/{timestamp}.tsv'
    
    # Initialize log file with headers
    log_file = open(log_filename, 'w')
    headers = [
        'timestamp', 'phase', 'epoch', 'batch', 'addend1', 'operator', 'addend2',
        'target', 'predicted', 'confidence', 'used_finger_counting', 'loss',
        'confidence_criterion'
    ]
    log_file.write('\t'.join(headers) + '\n')
    log_file.flush()
    
    print(f"Logging to: {log_filename}")
    return log_file

def log_training_step(log_file, phase: str, epoch: int, batch: int, 
                     addend1: Optional[int], operator: str, addend2: Optional[int],
                     target: int, predicted: int, confidence: float, 
                     used_finger_counting: bool, loss: float, confidence_criterion: float):
    """Log a training step"""
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
    
    # Convert None values to empty strings for logging
    a1_str = str(addend1) if addend1 is not None else ''
    a2_str = str(addend2) if addend2 is not None else ''
    
    log_entry = [
        timestamp, phase, str(epoch), str(batch), a1_str, operator, a2_str,
        str(target), str(predicted), f'{confidence:.4f}', 
        str(used_finger_counting), f'{loss:.6f}', f'{confidence_criterion:.3f}'
    ]
    
    log_file.write('\t'.join(log_entry) + '\n')
    log_file.flush()

# Encoding functions
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

# Activation functions
def softmax(x: np.ndarray) -> np.ndarray:
    """Softmax activation function"""
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

def relu(x: np.ndarray) -> np.ndarray:
    """ReLU activation function"""
    return np.maximum(0, x)

# Confidence calculation
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

# Data generation functions
def generate_counting_data(max_number: int = 12) -> List[Tuple[np.ndarray, int, Tuple]]:
    """Generate pre-training data for counting sequences"""
    data = []
    
    # Single number counting: "n -> n+1"
    for n in range(1, max_number):
        if n <= 5:  # Only input numbers 1-5
            x = encode_input(n, '->', None)
            y = n + 1
            details = (n, '->', None)
            data.append((x, y, details))
    
    # Two number counting: "n, n+1 -> n+2"
    for n in range(1, max_number - 1):
        if n <= 5 and n + 1 <= 5:  # Both numbers must be in range
            x = encode_input(n, '->', n + 1)
            y = n + 2
            details = (n, '->', n + 1)
            data.append((x, y, details))
    
    return data

def generate_addition_data() -> List[Tuple[np.ndarray, int, Tuple]]:
    """Generate training data for addition problems"""
    data = []
    for a1 in range(1, 6):  # 1-5
        for a2 in range(1, 6):  # 1-5
            x = encode_input(a1, '+', a2)
            y = a1 + a2
            details = (a1, '+', a2)
            data.append((x, y, details))
    return data

# Finger counting fallback
def finger_counting(addend1: int, addend2: int) -> int:
    """External mechanical method for getting addition answers"""
    return addend1 + addend2

class SMM:
    """Small Math Model - Neural network weights and forward/backward pass"""
    
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
        
        # Confidence criterion
        self.confidence_criterion = 0.5
    
    def forward(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Forward pass through the network"""
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
        x = encode_input(addend1, operator, addend2).reshape(1, -1)
        output_probs, _, attention = self.forward(x)
        
        confidence = calculate_confidence(output_probs[0], self.output_size)
        predicted_idx = np.argmax(output_probs[0])
        predicted_value = predicted_idx + 1  # Convert back to 1-12 range
        
        use_finger_counting = confidence < self.confidence_criterion
        
        return predicted_value, confidence, use_finger_counting
    
    def train_batch(self, X_batch: np.ndarray, y_batch: np.ndarray, 
                   log_file, phase: str = "training", epoch: int = 0, batch_num: int = 0,
                   input_details: List[Tuple] = None):
        """Train on a batch of data using backpropagation"""
        batch_size = X_batch.shape[0]
        
        # Forward pass
        output_probs, hidden_activations, attention_scores = self.forward(X_batch)
        
        # Convert labels to one-hot
        y_one_hot = np.zeros((batch_size, self.output_size))
        for i, label in enumerate(y_batch):
            if 1 <= label <= 12:
                y_one_hot[i, label - 1] = 1.0
        
        # Backward pass
        # Output layer gradients
        dZ2 = output_probs - y_one_hot
        dW2 = hidden_activations.T @ dZ2 / batch_size
        db2 = np.mean(dZ2, axis=0, keepdims=True)
        
        # Hidden layer gradients
        dA1 = dZ2 @ self.W2.T
        dZ1 = dA1 * (hidden_activations > 0)  # ReLU derivative
        dW1 = X_batch.T @ dZ1 / batch_size
        db1 = np.mean(dZ1, axis=0, keepdims=True)
        
        # Update weights
        self.W2 -= self.learning_rate * dW2
        self.b2 -= self.learning_rate * db2
        self.W1 -= self.learning_rate * dW1
        self.b1 -= self.learning_rate * db1
        
        # Calculate loss
        loss = -np.mean(np.sum(y_one_hot * np.log(output_probs + 1e-10), axis=1))
        
        # Log each sample in the batch
        if input_details:
            for i, (addend1, operator, addend2) in enumerate(input_details):
                predicted = np.argmax(output_probs[i]) + 1
                confidence = calculate_confidence(output_probs[i], self.output_size)
                used_finger_counting = confidence < self.confidence_criterion
                
                log_training_step(
                    log_file, phase, epoch, batch_num, addend1, operator, addend2,
                    y_batch[i], predicted, confidence, used_finger_counting, loss,
                    self.confidence_criterion
                )
        
        return loss

# Example usage
if __name__ == "__main__":
    # Setup logging
    log_file = setup_logging()
    
    # Initialize the model
    smm = SMM(hidden_size=64, learning_rate=0.01)
    
    # Test encoding
    print("Testing encodings:")
    print("Number 3:", encode_number(3))
    print("Operator '+':", encode_operator('+'))
    print("Operator '->':", encode_operator('->'))
    print("Input '2 + 3':", encode_input(2, '+', 3))
    print("Input '3 -> ?':", encode_input(3, '->', None))
    
    # Test prediction (before training)
    print("\nTesting prediction before training:")
    pred, conf, finger = smm.predict(2, '+', 3)
    print(f"2 + 3 = {pred}, confidence: {conf:.3f}, use finger counting: {finger}")
    
    # Generate training data
    counting_data = generate_counting_data()
    addition_data = generate_addition_data()
    
    print(f"\nGenerated {len(counting_data)} counting examples")
    print(f"Generated {len(addition_data)} addition examples")
    
    # Close log file
    log_file.close()

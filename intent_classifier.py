"""
PyTorch Intent Classifier - Agentic Router
A specialized classifier for routing simple queries to appropriate tools/services.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import random
import numpy as np
import time
import argparse

# Import training data
from data import TRAINING_DATA

# Set random seeds for reproducibility
torch.manual_seed(42)
random.seed(42)
np.random.seed(42)

# ============================================================================
# PREPROCESSING FUNCTIONS
# ============================================================================

def build_vocab(data):
    """
    Build vocabulary from training data.
    
    Returns:
        word_to_index: dict mapping words to indices
        intent_to_index: dict mapping intent labels to indices
        vocab_size: total number of unique words + special tokens
        num_classes: number of intent classes
    """
    # Extract all unique words
    all_words = set()
    all_intents = set()
    
    for text, intent in data:
        words = text.lower().split()
        all_words.update(words)
        all_intents.add(intent)
    
    # Create word to index mapping (0 for PAD, 1 for UNK)
    word_to_index = {"<PAD>": 0, "<UNK>": 1}
    for i, word in enumerate(sorted(all_words), start=2):
        word_to_index[word] = i
    
    # Create intent to index mapping
    intent_to_index = {intent: i for i, intent in enumerate(sorted(all_intents))}
    
    vocab_size = len(word_to_index)
    num_classes = len(intent_to_index)
    
    return word_to_index, intent_to_index, vocab_size, num_classes


def tokenize_and_pad(sentence, word_to_index, max_len):
    """
    Convert sentence to padded tensor of word indices.
    
    Args:
        sentence: input text string
        word_to_index: vocabulary mapping
        max_len: maximum sequence length
    
    Returns:
        tensor of shape [max_len] with word indices
    """
    words = sentence.lower().split()
    
    # Convert words to indices (use UNK for unknown words)
    indices = []
    for word in words:
        if word in word_to_index:
            indices.append(word_to_index[word])
        else:
            indices.append(word_to_index["<UNK>"])
    
    # Pad or truncate to max_len
    if len(indices) < max_len:
        indices += [word_to_index["<PAD>"]] * (max_len - len(indices))
    else:
        indices = indices[:max_len]
    
    return torch.tensor(indices, dtype=torch.long)


# ============================================================================
# DATASET CLASS
# ============================================================================

class IntentDataset(Dataset):
    """Custom Dataset for intent classification."""
    
    def __init__(self, data, word_to_index, intent_to_index, max_len):
        """
        Args:
            data: list of (text, intent) tuples
            word_to_index: vocabulary mapping
            intent_to_index: intent label mapping
            max_len: maximum sequence length
        """
        self.data = data
        self.word_to_index = word_to_index
        self.intent_to_index = intent_to_index
        self.max_len = max_len
        
        # Preprocess all data
        self.inputs = []
        self.labels = []
        
        for text, intent in data:
            input_tensor = tokenize_and_pad(text, word_to_index, max_len)
            label_index = intent_to_index[intent]
            self.inputs.append(input_tensor)
            self.labels.append(label_index)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.inputs[idx], self.labels[idx]


# ============================================================================
# MODEL ARCHITECTURE
# ============================================================================

class SimpleIntentClassifier(nn.Module):
    """
    Simple Intent Classifier with:
    - Embedding layer
    - Mean pooling (Bag of Embeddings)
    - Feed-forward network (FFN) classifier
    """
    
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_classes):
        """
        Args:
            vocab_size: size of vocabulary
            embedding_dim: dimension of word embeddings
            hidden_dim: dimension of hidden layer
            num_classes: number of intent classes
        """
        super(SimpleIntentClassifier, self).__init__()
        
        # Embedding layer (with padding_idx=0)
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=0
        )
        
        # Feed-forward network
        self.fc1 = nn.Linear(embedding_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(hidden_dim, num_classes)
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: tensor of shape [batch_size, seq_len]
        
        Returns:
            logits of shape [batch_size, num_classes]
        """
        # Embedding: [batch_size, seq_len] -> [batch_size, seq_len, embedding_dim]
        embedded = self.embedding(x)
        
        # Mean pooling: [batch_size, seq_len, embedding_dim] -> [batch_size, embedding_dim]
        pooled = embedded.mean(dim=1)
        
        # Feed-forward network
        hidden = self.fc1(pooled)
        hidden = self.relu(hidden)
        hidden = self.dropout(hidden)
        output = self.fc2(hidden)
        
        return output


# ============================================================================
# INFERENCE FUNCTION
# ============================================================================

def predict_intent(sentence, model, word_to_index, max_len, index_to_intent):
    """
    Predict intent for a given sentence.
    
    Args:
        sentence: input text string
        model: trained model
        word_to_index: vocabulary mapping
        max_len: maximum sequence length
        index_to_intent: mapping from index to intent label
    
    Returns:
        predicted_intent: intent label (string)
        confidence: confidence score (float)
    """
    model.eval()
    
    with torch.no_grad():
        # Preprocess input
        input_tensor = tokenize_and_pad(sentence, word_to_index, max_len)
        input_tensor = input_tensor.unsqueeze(0)  # Add batch dimension: [1, max_len]
        
        # Forward pass
        logits = model(input_tensor)  # [1, num_classes]
        
        # Get probabilities
        probs = torch.softmax(logits, dim=1)
        
        # Get prediction
        confidence, predicted_idx = torch.max(probs, dim=1)
        
        # Convert to intent label
        intent = index_to_intent[predicted_idx.item()]
        
        return intent, confidence.item()


# ============================================================================
# MAIN TRAINING & DEMONSTRATION
# ============================================================================

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='PyTorch Intent Classifier - Agentic Router')
    parser.add_argument(
        '--input',
        type=str,
        help='Custom sentence to classify (e.g., --input "what is the weather")'
    )
    args = parser.parse_args()
    
    print("=" * 70)
    print("PyTorch Intent Classifier - Agentic Router")
    print("=" * 70)
    
    # Hyperparameters
    EMBEDDING_DIM = 50
    HIDDEN_DIM = 32
    EPOCHS = 100  # Increased from 50
    LEARNING_RATE = 0.001
    BATCH_SIZE = 4
    CONFIDENCE_THRESHOLD = 0.8
    
    print(f"\nHyperparameters:")
    print(f"  Embedding Dim: {EMBEDDING_DIM}")
    print(f"  Hidden Dim: {HIDDEN_DIM}")
    print(f"  Epochs: {EPOCHS}")
    print(f"  Learning Rate: {LEARNING_RATE}")
    print(f"  Batch Size: {BATCH_SIZE}")
    print(f"  Confidence Threshold: {CONFIDENCE_THRESHOLD}")
    
    # Build vocabulary
    print(f"\nBuilding vocabulary from {len(TRAINING_DATA)} training examples...")
    word_to_index, intent_to_index, vocab_size, num_classes = build_vocab(TRAINING_DATA)
    
    # Calculate max sequence length
    max_seq_len = max(len(text.split()) for text, _ in TRAINING_DATA)
    
    print(f"  Vocabulary Size: {vocab_size}")
    print(f"  Number of Classes: {num_classes}")
    print(f"  Max Sequence Length: {max_seq_len}")
    print(f"  Intent Classes: {list(intent_to_index.keys())}")
    
    # Create reverse mapping for inference
    index_to_intent = {v: k for k, v in intent_to_index.items()}
    
    # Create dataset and dataloader
    dataset = IntentDataset(TRAINING_DATA, word_to_index, intent_to_index, max_seq_len)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    # Initialize model, loss, optimizer
    model = SimpleIntentClassifier(
        vocab_size=vocab_size,
        embedding_dim=EMBEDDING_DIM,
        hidden_dim=HIDDEN_DIM,
        num_classes=num_classes
    )
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    print(f"\nModel Architecture:")
    print(f"  Total Parameters: {sum(p.numel() for p in model.parameters())}")
    print(model)
    
    # Training loop
    print(f"\n{'=' * 70}")
    print("TRAINING")
    print("=" * 70)
    
    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0
        
        for inputs, labels in dataloader:
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        # Print progress every 10 epochs
        if (epoch + 1) % 10 == 0:
            avg_loss = total_loss / len(dataloader)
            print(f"Epoch [{epoch + 1}/{EPOCHS}], Loss: {avg_loss:.4f}")
    
    print("\nTraining complete!")
    
    # Demonstration with test sentences or custom input
    print(f"\n{'=' * 70}")
    
    # Check if custom input provided
    if args.input:
        print("CUSTOM INPUT CLASSIFICATION")
        print("=" * 70)
        
        # Classify the custom input
        start_time = time.time()
        intent, confidence = predict_intent(
            args.input, model, word_to_index, max_seq_len, index_to_intent
        )
        end_time = time.time()
        
        latency_ms = (end_time - start_time) * 1000
        
        print(f"\nInput: \"{args.input}\"")
        print(f"Predicted Intent: {intent}")
        print(f"Confidence: {confidence:.4f}")
        print(f"Latency: {latency_ms:.2f} ms")
        print(f"\nConfidence Threshold: {CONFIDENCE_THRESHOLD}")
        
        # Agentic routing decision
        if confidence > CONFIDENCE_THRESHOLD:
            print(f"\n✓ FAST PATH: Executing tool: {intent}")
            print(f"  Action: Route to {intent} service")
        else:
            print(f"\n⚠ COSTLY PATH: Intent too vague, Fallback to General LLM")
            print(f"  Action: Route to general-purpose LLM for handling")
        
        print("\n" + "=" * 70)
    
    else:
        # Run full demo with test sentences
        print("INFERENCE DEMONSTRATION - Agentic Routing")
        print("=" * 70)
        
        test_sentences = [
            "hi",
            "greetings",
            "what's the temperature",
            "how is the weather today",
            "I need a place to stay",
            "book me a ticket",
            "thanks",
            "appreciate it",
            "current time",
            "goodbye",
            "see ya",
            "what is the weather forecast",
            "random query that makes no sense xyz",
        ]
        
        print(f"\nConfidence Threshold: {CONFIDENCE_THRESHOLD}")
        print(f"  > {CONFIDENCE_THRESHOLD}: Fast Path (Execute Tool)")
        print(f"  ≤ {CONFIDENCE_THRESHOLD}: Costly Path (Fallback to General LLM)\n")
        
        total_latency = 0
        
        for sentence in test_sentences:
            # Measure inference latency
            start_time = time.time()
            intent, confidence = predict_intent(
                sentence, model, word_to_index, max_seq_len, index_to_intent
            )
            end_time = time.time()
            
            latency_ms = (end_time - start_time) * 1000  # Convert to milliseconds
            total_latency += latency_ms
            
            print("-" * 70)
            print(f"Input: \"{sentence}\"")
            print(f"Predicted Intent: {intent}")
            print(f"Confidence: {confidence:.4f}")
            print(f"Latency: {latency_ms:.2f} ms")
            
            # Agentic routing decision
            if confidence > CONFIDENCE_THRESHOLD:
                print(f"✓ FAST PATH: Executing tool: {intent}")
            else:
                print(f"⚠ COSTLY PATH: Intent too vague, Fallback to General LLM")
        
        print("=" * 70)
        
        # Latency summary
        avg_latency = total_latency / len(test_sentences)
        print(f"\nLatency Summary:")
        print(f"  Total Inference Time: {total_latency:.2f} ms")
        print(f"  Average Latency per Query: {avg_latency:.2f} ms")
        print(f"  Queries Processed: {len(test_sentences)}")
        print(f"\nComparison:")
        print(f"  Intent Classifier: ~{avg_latency:.1f} ms")
        print(f"  Typical LLM API Call: ~1000-2000 ms")
        print(f"  Speed Improvement: ~{1500/avg_latency:.1f}x faster")
        
        print("=" * 70)
    
    print("Demo complete!")
    print("=" * 70)
    
    # Interactive mode - accept additional custom inputs
    print("\n" + "=" * 70)
    print("INTERACTIVE MODE")
    print("=" * 70)
    print("\nYou can now enter custom sentences for classification.")
    print("Type 'quit' or 'exit' to end the session.\n")
    
    while True:
        try:
            # Get user input
            user_input = input("Enter a sentence: ").strip()
            
            # Check for exit commands
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("\nExiting interactive mode. Goodbye!")
                break
            
            # Skip empty inputs
            if not user_input:
                continue
            
            # Classify the input
            start_time = time.time()
            intent, confidence = predict_intent(
                user_input, model, word_to_index, max_seq_len, index_to_intent
            )
            end_time = time.time()
            
            latency_ms = (end_time - start_time) * 1000
            
            # Display results
            print("\n" + "-" * 70)
            print(f"Input: \"{user_input}\"")
            print(f"Predicted Intent: {intent}")
            print(f"Confidence: {confidence:.4f}")
            print(f"Latency: {latency_ms:.2f} ms")
            
            # Routing decision
            if confidence > CONFIDENCE_THRESHOLD:
                print(f"✓ FAST PATH: Executing tool: {intent}")
            else:
                print(f"⚠ COSTLY PATH: Intent too vague, Fallback to General LLM")
            print("-" * 70 + "\n")
            
        except KeyboardInterrupt:
            print("\n\nInterrupted. Exiting interactive mode. Goodbye!")
            break
        except EOFError:
            print("\n\nEnd of input. Exiting interactive mode. Goodbye!")
            break
    
    print("\n" + "=" * 70)
    print("Session complete!")
    print("=" * 70)


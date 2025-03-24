import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import T5ForConditionalGeneration, T5Tokenizer
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
import time
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist


class SymbolSpace:
    """
    Represents the symbolic layer using Gaussian Mixture Models at multiple scales.
    """
    def __init__(self, 
                 n_components_initial: int = 10, 
                 n_dims_reduced: int = 50,
                 layer_scales: List[int] = [2, 6, 11],  # Sample from low, mid, high layers
                 min_symbol_weight: float = 0.02,
                 merge_threshold_kl: float = 2.0):
        """
        Initialize the symbol space.
        
        Args:
            n_components_initial: Initial number of GMM components
            n_dims_reduced: Dimensionality after PCA reduction
            layer_scales: Which transformer layers to sample for different scales
            min_symbol_weight: Minimum mixing coefficient to keep a symbol
            merge_threshold_kl: KL divergence threshold for merging symbols
        """
        self.n_components_initial = n_components_initial
        self.n_dims_reduced = n_dims_reduced
        self.layer_scales = layer_scales
        self.min_symbol_weight = min_symbol_weight
        self.merge_threshold_kl = merge_threshold_kl
        
        # Symbol spaces at different scales
        self.symbols = {layer: None for layer in layer_scales}
        
        # PCA reducers for each layer
        self.pca_reducers = {layer: None for layer in layer_scales}
        
        # Symbol-token connections (for attention guidance)
        self.symbol_token_connections = {layer: [] for layer in layer_scales}
        
        # Cross-scale relations
        self.symbol_relations = {}
        
        # Computational cache
        self.symbol_cache = {}
        
    def fit_pca(self, layer_activations: Dict[int, List[np.ndarray]]):
        """
        Fit PCA reducers on activation data from each layer.
        
        Args:
            layer_activations: Dictionary mapping layer indices to lists of activation matrices
        """
        for layer in self.layer_scales:
            if layer in layer_activations and layer_activations[layer]:
                # Concatenate all activation matrices for this layer
                all_activations = np.vstack([act.reshape(-1, act.shape[-1]) for act in layer_activations[layer]])
                
                # Fit PCA
                self.pca_reducers[layer] = PCA(n_components=self.n_dims_reduced)
                self.pca_reducers[layer].fit(all_activations)
                print(f"Fitted PCA for layer {layer}, explained variance: "
                      f"{np.sum(self.pca_reducers[layer].explained_variance_ratio_):.4f}")
    
    def fit_symbols(self, layer_activations: Dict[int, List[np.ndarray]]):
        """
        Fit GMMs on reduced activation data from each layer.
        
        Args:
            layer_activations: Dictionary mapping layer indices to lists of activation matrices
        """
        for layer in self.layer_scales:
            if layer in layer_activations and layer_activations[layer] and self.pca_reducers[layer] is not None:
                # Concatenate and reduce activations
                all_activations = np.vstack([act.reshape(-1, act.shape[-1]) for act in layer_activations[layer]])
                reduced_activations = self.pca_reducers[layer].transform(all_activations)
                
                # Fit GMM
                self.symbols[layer] = GaussianMixture(
                    n_components=self.n_components_initial,
                    covariance_type='full',
                    random_state=42
                )
                self.symbols[layer].fit(reduced_activations)
                print(f"Fitted GMM for layer {layer} with {self.n_components_initial} components")
                
                # Store original shapes for reconstruction
                self._record_original_shapes(layer_activations[layer])
    
    def _record_original_shapes(self, activations: List[np.ndarray]):
        """Record original shapes of activation tensors for later reconstruction."""
        self.original_shapes = [act.shape for act in activations]
    
    def encode_symbols(self, 
                       layer_activations: Dict[int, torch.Tensor], 
                       input_tokens: List[int]) -> Dict[int, np.ndarray]:
        """
        Encode input activations into symbols and track token-symbol connections.
        
        Args:
            layer_activations: Dictionary mapping layer indices to activation tensors
            input_tokens: List of input token ids
            
        Returns:
            Dictionary mapping layer indices to symbol assignment probabilities
        """
        symbol_probs = {}
        
        for layer in self.layer_scales:
            if layer in layer_activations and self.symbols[layer] is not None:
                # Get activations and reduce dimensionality
                activations = layer_activations[layer].detach().cpu().numpy()
                batch_size, seq_len, hidden_dim = activations.shape
                activations_flat = activations.reshape(-1, hidden_dim)
                reduced_activations = self.pca_reducers[layer].transform(activations_flat)
                
                # Get symbol assignments
                probs = self.symbols[layer].predict_proba(reduced_activations)
                symbol_probs[layer] = probs.reshape(batch_size, seq_len, -1)
                
                # Record token-symbol connections
                self._update_token_symbol_connections(layer, probs, input_tokens, batch_size, seq_len)
                
                # Cache symbol activations for computational shortcutting
                self._cache_symbol_activations(layer, probs, activations, batch_size, seq_len)
        
        return symbol_probs
    
    def _update_token_symbol_connections(self, 
                                         layer: int, 
                                         probs: np.ndarray, 
                                         input_tokens: List[int],
                                         batch_size: int, 
                                         seq_len: int):
        """
        Update the token-symbol connection matrix.
        
        Args:
            layer: Layer index
            probs: Symbol assignment probabilities
            input_tokens: Input token ids
            batch_size: Batch size
            seq_len: Sequence length
        """
        probs_reshaped = probs.reshape(batch_size, seq_len, -1)
        
        # For each sequence in the batch
        for b in range(batch_size):
            # Get the sequence tokens
            sequence_tokens = input_tokens[b * seq_len:(b + 1) * seq_len]
            
            # For each token position
            for pos in range(seq_len):
                token_id = sequence_tokens[pos]
                symbol_assignments = probs_reshaped[b, pos]
                
                # Record connections for symbols with probability > 0.2
                for symbol_idx, prob in enumerate(symbol_assignments):
                    if prob > 0.2:  # Only record strong connections
                        self.symbol_token_connections[layer].append({
                            'token_id': token_id,
                            'position': pos,
                            'batch': b,
                            'symbol_idx': symbol_idx,
                            'probability': prob
                        })
    
    def _cache_symbol_activations(self, 
                                 layer: int, 
                                 probs: np.ndarray, 
                                 activations: np.ndarray,
                                 batch_size: int, 
                                 seq_len: int):
        """
        Cache activation patterns associated with symbols for computational shortcutting.
        
        Args:
            layer: Layer index
            probs: Symbol assignment probabilities
            activations: Raw activation values
            batch_size: Batch size
            seq_len: Sequence length
        """
        # Reshape for per-token analysis
        probs_reshaped = probs.reshape(batch_size, seq_len, -1)
        activations_reshaped = activations.reshape(batch_size, seq_len, -1)
        
        # For each symbol
        for symbol_idx in range(probs_reshaped.shape[-1]):
            # Get all activations strongly assigned to this symbol
            strong_matches = []
            
            for b in range(batch_size):
                for pos in range(seq_len):
                    if probs_reshaped[b, pos, symbol_idx] > 0.7:  # Strong association
                        strong_matches.append(activations_reshaped[b, pos])
            
            # If we have enough examples, compute an average activation pattern
            if strong_matches:
                avg_activation = np.mean(strong_matches, axis=0)
                if layer not in self.symbol_cache:
                    self.symbol_cache[layer] = {}
                self.symbol_cache[layer][symbol_idx] = avg_activation
    
    def generate_attention_bias(self, 
                               layer: int, 
                               current_symbols: np.ndarray, 
                               batch_size: int, 
                               seq_len: int) -> np.ndarray:
        """
        Generate attention bias based on symbol-token connections.
        
        Args:
            layer: Layer index to use for guidance
            current_symbols: Current decoder symbol activations
            batch_size: Batch size
            seq_len: Input sequence length
            
        Returns:
            Attention bias matrix to add to cross-attention scores
        """
        # Default bias is zero
        bias = np.zeros((batch_size, seq_len))
        
        # If we don't have connections for this layer, return default bias
        if not self.symbol_token_connections[layer]:
            return bias
        
        # Get the most active symbol for each position in current_symbols
        active_symbols = np.argmax(current_symbols, axis=-1)
        
        # For each batch item
        for b in range(batch_size):
            # Get the active symbols for this batch
            batch_symbols = active_symbols[b]
            
            # Find token connections for these symbols
            for symbol_idx in batch_symbols:
                relevant_connections = [
                    conn for conn in self.symbol_token_connections[layer]
                    if conn['symbol_idx'] == symbol_idx and conn['batch'] == b
                ]
                
                # Add bias based on connection strength
                for conn in relevant_connections:
                    bias[b, conn['position']] += conn['probability']
        
        # Normalize bias
        if np.max(bias) > 0:
            bias = bias / np.max(bias)
        
        return bias
    
    def shortcut_computation(self, 
                            layer: int, 
                            activations: torch.Tensor, 
                            symbol_probs: np.ndarray,
                            threshold: float = 0.8) -> Tuple[torch.Tensor, bool]:
        """
        Attempt to shortcut computation by reusing cached symbol activations.
        
        Args:
            layer: Layer index
            activations: Current activation tensor
            symbol_probs: Current symbol assignment probabilities
            threshold: Probability threshold for using cache
            
        Returns:
            (Potentially modified activation tensor, whether shortcut was used)
        """
        # If we have no cache for this layer, no shortcut possible
        if layer not in self.symbol_cache:
            return activations, False
        
        # Convert to numpy for processing
        activations_np = activations.detach().cpu().numpy()
        batch_size, seq_len, hidden_dim = activations_np.shape
        activations_modified = activations_np.copy()
        shortcut_used = False
        
        # For each sequence position
        for b in range(batch_size):
            for pos in range(seq_len):
                # Get the most probable symbol and its probability
                symbol_idx = np.argmax(symbol_probs[b, pos])
                prob = symbol_probs[b, pos, symbol_idx]
                
                # If probability is high enough and we have this symbol cached
                if prob > threshold and symbol_idx in self.symbol_cache[layer]:
                    # Use the cached activation
                    activations_modified[b, pos] = self.symbol_cache[layer][symbol_idx]
                    shortcut_used = True
        
        # Convert back to tensor
        if shortcut_used:
            return torch.tensor(activations_modified, device=activations.device), True
        else:
            return activations, False
    
    def prune_and_merge_symbols(self, min_weight: Optional[float] = None):
        """
        Prune low-weight symbols and merge similar symbols.
        
        Args:
            min_weight: Optional override for minimum symbol weight
        """
        if min_weight is None:
            min_weight = self.min_symbol_weight
        
        for layer in self.layer_scales:
            if self.symbols[layer] is not None:
                gmm = self.symbols[layer]
                
                # Identify low-weight components
                low_weight_idx = np.where(gmm.weights_ < min_weight)[0]
                
                # TODO: Implement symbol merging based on KL divergence
                # This is a placeholder for more sophisticated symbol management
                
                print(f"Layer {layer}: {len(low_weight_idx)} symbols below weight threshold")
                
                # For now, just report how many symbols would be pruned
                # In a full implementation, we would rebuild the GMM without these components
    
    def identify_cross_scale_relations(self):
        """
        Identify relationships between symbols at different scales.
        """
        # This is a placeholder for cross-scale relation identification
        # In a full implementation, we would:
        # 1. Check for co-activation patterns between scales
        # 2. Build directed graphs between symbols at different scales
        # 3. Track compositional relationships
        
        print("Cross-scale relation identification not yet implemented")


class MetasymbolicT5:
    """
    T5 model augmented with metasymbolic capabilities.
    """
    def __init__(self, 
                 model_name: str = 't5-small',
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        """
        Initialize the metasymbolic T5 model.
        
        Args:
            model_name: Name of the T5 model to use
            device: Device to run the model on
        """
        self.model_name = model_name
        self.device = device
        
        # Load model and tokenizer
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        self.model = T5ForConditionalGeneration.from_pretrained(model_name)
        self.model.to(device)
        
        # Initialize symbol space
        self.symbol_space = SymbolSpace()
        
        # Store model hooks
        self.hooks = []
        
        # Set up to collect activations
        self.activation_dict = {}
    
    def _activation_hook(self, name):
        """
        Create a hook function for collecting activations.
        
        Args:
            name: Name for the activations
            
        Returns:
            Hook function
        """
        def hook(module, input, output):
            self.activation_dict[name] = output.detach()
        return hook
    
    def _register_hooks(self):
        """Register hooks to capture activations."""
        # Remove existing hooks
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        
        # Register new hooks for encoder layers
        for layer_idx in self.symbol_space.layer_scales:
            # Adjust indices for T5's layer naming
            layer = self.model.encoder.block[layer_idx]
            hook = layer.register_forward_hook(self._activation_hook(f"encoder_layer_{layer_idx}"))
            self.hooks.append(hook)
    
    def collect_activations(self, dataloader: DataLoader, max_batches: int = 50) -> Dict[int, List[np.ndarray]]:
        """
        Collect activations from the model on a dataset.
        
        Args:
            dataloader: DataLoader for the dataset
            max_batches: Maximum number of batches to process
            
        Returns:
            Dictionary mapping layer indices to lists of activation matrices
        """
        print(f"Collecting activations on {max_batches} batches...")
        
        # Register hooks
        self._register_hooks()
        
        # Storage for activations
        layer_activations = {layer: [] for layer in self.symbol_space.layer_scales}
        
        # Process data
        self.model.eval()
        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                if batch_idx >= max_batches:
                    break
                
                # Clear activation dictionary
                self.activation_dict = {}
                
                # Get input_ids and masks
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                
                # Forward pass to trigger hooks
                _ = self.model(input_ids=input_ids, attention_mask=attention_mask)
                
                # Store activations
                for layer in self.symbol_space.layer_scales:
                    key = f"encoder_layer_{layer}"
                    if key in self.activation_dict:
                        layer_activations[layer].append(self.activation_dict[key].cpu().numpy())
        
        print("Activation collection complete.")
        return layer_activations
    
    def build_symbol_space(self, dataloader: DataLoader, max_batches: int = 50):
        """
        Build the symbol space from a dataset.
        
        Args:
            dataloader: DataLoader for the dataset
            max_batches: Maximum number of batches to process
        """
        # Collect activations
        layer_activations = self.collect_activations(dataloader, max_batches)
        
        # Fit PCA reducers
        self.symbol_space.fit_pca(layer_activations)
        
        # Fit symbol GMMs
        self.symbol_space.fit_symbols(layer_activations)
        
        # Prune and merge symbols
        self.symbol_space.prune_and_merge_symbols()
        
        # Identify cross-scale relations
        self.symbol_space.identify_cross_scale_relations()
    
    def _modify_cross_attention(self, attention_scores: torch.Tensor, layer_idx: int) -> torch.Tensor:
        """
        Modify cross-attention scores using symbol guidance.
        
        Args:
            attention_scores: Original attention scores [batch_size, num_heads, tgt_len, src_len]
            layer_idx: Decoder layer index
            
        Returns:
            Modified attention scores
        """
        batch_size, num_heads, tgt_len, src_len = attention_scores.shape
        
        # 1. Get current decoder activations
        decoder_key = f"decoder_layer_{layer_idx}"
        if decoder_key not in self.activation_dict:
            # If we don't have decoder activations, return unmodified scores
            return attention_scores
        
        decoder_activations = self.activation_dict[decoder_key]
        
        # Find corresponding encoder layer (assuming we want to use symbols from same scale)
        encoder_layer_idx = self.symbol_space.layer_scales[min(len(self.symbol_space.layer_scales) - 1, 
                                                           layer_idx // 2)]  # Simple mapping strategy
        
        # 2. Map decoder activations to symbols
        if self.symbol_space.symbols[encoder_layer_idx] is None or self.symbol_space.pca_reducers[encoder_layer_idx] is None:
            # No symbols available for this layer
            return attention_scores
        
        # Get activations and reduce dimensionality
        dec_activations_np = decoder_activations.detach().cpu().numpy()
        dec_batch_size, dec_seq_len, hidden_dim = dec_activations_np.shape
        dec_activations_flat = dec_activations_np.reshape(-1, hidden_dim)
        
        # Apply PCA reduction
        reduced_activations = self.symbol_space.pca_reducers[encoder_layer_idx].transform(dec_activations_flat)
        
        # Get symbol assignments
        symbol_probs = self.symbol_space.symbols[encoder_layer_idx].predict_proba(reduced_activations)
        symbol_probs = symbol_probs.reshape(dec_batch_size, dec_seq_len, -1)
        
        # 3. Generate attention bias from symbol-token connections
        attention_bias = self.symbol_space.generate_attention_bias(
            encoder_layer_idx,
            symbol_probs,
            batch_size,
            src_len
        )
        
        # Convert to tensor and reshape for broadcasting across heads
        attention_bias = torch.tensor(
            attention_bias, 
            dtype=attention_scores.dtype, 
            device=attention_scores.device
        ).unsqueeze(1).unsqueeze(1)  # [batch_size, 1, 1, src_len]
        
        # 4. Add bias to attention scores
        # Scale bias (typically 0.1 to 1.0 depending on how strong you want the influence to be)
        bias_scale = 0.3
        scaled_bias = attention_bias * bias_scale
        
        # Add to original scores
        modified_scores = attention_scores + scaled_bias
        
        # Log statistics about modification (in a real implementation)
        # avg_change = (modified_scores - attention_scores).abs().mean().item()
        # print(f"Average attention score change: {avg_change:.4f}")
        
        return modified_scores
    
    def generate(self, 
                input_ids: torch.Tensor, 
                attention_mask: torch.Tensor, 
                max_length: int = 64,
                use_symbols: bool = True,
                **kwargs) -> Tuple[torch.Tensor, Dict]:
        """
        Generate text with the option to use symbolic guidance.
        
        Args:
            input_ids: Input token ids
            attention_mask: Attention mask
            max_length: Maximum generation length
            use_symbols: Whether to use symbolic guidance
            **kwargs: Additional arguments for generation
            
        Returns:
            (Generated token ids, Dictionary with statistics)
        """
        # Statistics
        stats = {
            'time_with_symbols': 0,
            'time_without_symbols': 0,
            'shortcut_count': 0,
            'total_steps': 0
        }
        
        # First, run without symbols to compare
        self.model.eval()
        with torch.no_grad():
            # Time standard generation
            start_time = time.time()
            standard_outputs = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=max_length,
                **kwargs
            )
            stats['time_without_symbols'] = time.time() - start_time
        
        # If not using symbols, return standard outputs
        if not use_symbols:
            return standard_outputs, stats
        
        # For symbol-guided generation, we need to:
        # 1. Process the encoder input to build symbol-token connections
        # 2. Add hooks to intercept and modify cross-attention
        # 3. Run generation with these hooks active
        
        # First, encode the input and build symbol connections
        self._register_hooks()
        
        # Run encoder to build symbol space for this input
        encoder_outputs = self.model.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        
        # Extract and encode symbols from activation patterns
        encoder_layer_activations = {}
        for layer in self.symbol_space.layer_scales:
            key = f"encoder_layer_{layer}"
            if key in self.activation_dict:
                encoder_layer_activations[layer] = self.activation_dict[key]
        
        # Update symbol-token connections
        for layer, activations in encoder_layer_activations.items():
            if self.symbol_space.symbols[layer] is not None:
                # Get reduced activations
                activations_np = activations.detach().cpu().numpy()
                batch_size, seq_len, hidden_dim = activations_np.shape
                activations_flat = activations_np.reshape(-1, hidden_dim)
                reduced_activations = self.symbol_space.pca_reducers[layer].transform(activations_flat)
                
                # Get symbol probabilities
                probs = self.symbol_space.symbols[layer].predict_proba(reduced_activations)
                reshaped_probs = probs.reshape(batch_size, seq_len, -1)
                
                # Update token-symbol connections
                input_tokens = input_ids.reshape(-1).cpu().tolist()
                self.symbol_space.encode_symbols({layer: activations}, input_tokens)
        
        # Register hooks for cross-attention modification
        cross_attention_hooks = []
        
        def modify_cross_attention_hook(layer_idx):
            def hook(module, inputs, outputs):
                # The outputs are the attention scores before softmax
                modified_scores = self._modify_cross_attention(outputs, layer_idx)
                return modified_scores
            return hook
        
        # Add hooks to cross-attention modules in the decoder
        for layer_idx, layer in enumerate(self.model.decoder.block):
            # Find the cross-attention module
            cross_attn = layer.layer[1].EncDecAttention.q
            # Register hook
            hook = cross_attn.register_forward_hook(modify_cross_attention_hook(layer_idx))
            cross_attention_hooks.append(hook)
        
        try:
            # Time generation with symbolic guidance
            start_time = time.time()
            
            # Generate with cross-attention modification
            symbolic_outputs = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                encoder_outputs=encoder_outputs,
                max_length=max_length,
                **kwargs
            )
            
            stats['time_with_symbols'] = time.time() - start_time
            
            # Count shortcuts that were actually used
            shortcut_count = 0
            for layer in self.symbol_space.layer_scales:
                # This is an estimate based on cache usage
                if layer in self.activation_dict:
                    shortcut_count += len(self.symbol_space.symbol_cache.get(layer, {}))
            
            stats['shortcut_count'] = shortcut_count
            stats['total_steps'] = max_length
            
            return symbolic_outputs, stats
            
        finally:
            # Remove hooks
            for hook in cross_attention_hooks:
                hook.remove()
        
        return symbolic_outputs, stats
    
    def evaluate_efficiency(self, 
                           test_dataloader: DataLoader, 
                           max_batches: int = 10) -> Dict:
        """
        Evaluate efficiency gains from symbolic processing.
        
        Args:
            test_dataloader: DataLoader for test data
            max_batches: Maximum number of batches to process
            
        Returns:
            Dictionary with efficiency statistics
        """
        all_stats = {
            'time_with_symbols': [],
            'time_without_symbols': [],
            'shortcut_ratio': [],
            'speedup': []
        }
        
        for batch_idx, batch in enumerate(test_dataloader):
            if batch_idx >= max_batches:
                break
            
            # Get inputs
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            
            # Generate with and without symbols
            _, stats = self.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_symbols=True,
                max_length=64
            )
            
            # Record statistics
            all_stats['time_with_symbols'].append(stats['time_with_symbols'])
            all_stats['time_without_symbols'].append(stats['time_without_symbols'])
            all_stats['shortcut_ratio'].append(stats['shortcut_count'] / stats['total_steps'])
            all_stats['speedup'].append(stats['time_without_symbols'] / max(stats['time_with_symbols'], 1e-6))
        
        # Calculate averages
        results = {
            'avg_time_with_symbols': np.mean(all_stats['time_with_symbols']),
            'avg_time_without_symbols': np.mean(all_stats['time_without_symbols']),
            'avg_shortcut_ratio': np.mean(all_stats['shortcut_ratio']),
            'avg_speedup': np.mean(all_stats['speedup'])
        }
        
        return results
    
    def visualize_symbols(self, layer_idx: int):
        """
        Visualize the symbols at a given layer.
        
        Args:
            layer_idx: Layer index to visualize
        """
        if self.symbol_space.symbols[layer_idx] is None:
            print(f"No symbols available for layer {layer_idx}")
            return
        
        gmm = self.symbol_space.symbols[layer_idx]
        
        # Reduce to 2D for visualization
        pca_2d = PCA(n_components=2)
        means_2d = pca_2d.fit_transform(gmm.means_)
        
        # Plot centers
        plt.figure(figsize=(10, 8))
        sizes = gmm.weights_ * 1000  # Scale by mixing coefficient
        plt.scatter(means_2d[:, 0], means_2d[:, 1], s=sizes, alpha=0.6)
        
        # Plot component indices
        for i, (x, y) in enumerate(means_2d):
            plt.text(x, y, str(i), fontsize=9)
        
        plt.title(f"Symbol space for layer {layer_idx}")
        plt.xlabel("Principal Component 1")
        plt.ylabel("Principal Component 2")
        plt.grid(True, alpha=0.3)
        plt.show()


# Example usage
class MathDataset(Dataset):
    """Example dataset for mathematical reasoning tasks."""
    def __init__(self, tokenizer, num_examples=1000, max_length=64):
        self.tokenizer = tokenizer
        self.num_examples = num_examples
        self.max_length = max_length
        self.examples = self._generate_examples()
    
    def _generate_examples(self):
        """Generate simple math examples."""
        examples = []
        for i in range(self.num_examples):
            # Simple addition problems
            a = np.random.randint(1, 100)
            b = np.random.randint(1, 100)
            input_text = f"Calculate: {a} + {b} = "
            output_text = str(a + b)
            examples.append((input_text, output_text))
        return examples
    
    def __len__(self):
        return self.num_examples
    
    def __getitem__(self, idx):
        input_text, output_text = self.examples[idx]
        
        # Tokenize
        input_encoding = self.tokenizer(
            input_text, 
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        output_encoding = self.tokenizer(
            output_text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        return {
            "input_ids": input_encoding.input_ids.squeeze(),
            "attention_mask": input_encoding.attention_mask.squeeze(),
            "labels": output_encoding.input_ids.squeeze(),
            "decoder_attention_mask": output_encoding.attention_mask.squeeze()
        }


def main():
    # Initialize model
    metasymbolic_t5 = MetasymbolicT5(model_name='t5-small')
    
    # Create dataset and dataloader
    dataset = MathDataset(tokenizer=metasymbolic_t5.tokenizer, num_examples=200)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
    
    # Build symbol space
    metasymbolic_t5.build_symbol_space(dataloader, max_batches=10)
    
    # Evaluate efficiency
    efficiency_stats = metasymbolic_t5.evaluate_efficiency(dataloader, max_batches=5)
    
    # Print results
    print("\nEfficiency Results:")
    for key, value in efficiency_stats.items():
        print(f"{key}: {value}")
    
    # Visualize symbols from one layer
    metasymbolic_t5.visualize_symbols(layer_idx=6)  # Middle layer


if __name__ == "__main__":
    main()
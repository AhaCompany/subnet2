"""
Optimizations for the Omron miner to improve performance and scoring.
"""

import os
import time
import hashlib
import threading
import functools
import gc
import psutil
from typing import Dict, Any, Tuple, List, Optional, Set

import bittensor as bt
import torch

try:
    import numpy as np
except ImportError:
    pass

# Maximum size for the proof cache
MAX_CACHE_SIZE = 500

# Time threshold for cache cleanup (in seconds)
CACHE_CLEANUP_THRESHOLD = 3600  # 1 hour


class ProofCache:
    """A thread-safe cache for storing and retrieving proofs."""
    
    def __init__(self, max_size: int = MAX_CACHE_SIZE):
        self.max_size = max_size
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.last_accessed: Dict[str, float] = {}
        self.lock = threading.RLock()
        self.last_cleanup = time.time()
    
    def get(self, key: str) -> Optional[Dict[str, Any]]:
        """Get a proof from the cache."""
        with self.lock:
            if key in self.cache:
                self.last_accessed[key] = time.time()
                return self.cache[key]
            return None
    
    def set(self, key: str, value: Dict[str, Any]) -> None:
        """Add a proof to the cache."""
        with self.lock:
            # Check if we need to perform cleanup
            current_time = time.time()
            if current_time - self.last_cleanup > CACHE_CLEANUP_THRESHOLD:
                self._cleanup()
            
            # If cache is full, remove least recently used items
            if len(self.cache) >= self.max_size:
                self._evict_lru()
            
            self.cache[key] = value
            self.last_accessed[key] = current_time
    
    def _cleanup(self) -> None:
        """Clean up old entries in the cache."""
        current_time = time.time()
        keys_to_remove = []
        
        for key, last_access in self.last_accessed.items():
            if current_time - last_access > CACHE_CLEANUP_THRESHOLD:
                keys_to_remove.append(key)
        
        for key in keys_to_remove:
            del self.cache[key]
            del self.last_accessed[key]
        
        self.last_cleanup = current_time
    
    def _evict_lru(self) -> None:
        """Evict the least recently used items."""
        if not self.last_accessed:
            return
        
        # Get the least recently used key
        lru_key = min(self.last_accessed.items(), key=lambda x: x[1])[0]
        
        # Remove it from both dicts
        del self.cache[lru_key]
        del self.last_accessed[lru_key]
    
    def clear(self) -> None:
        """Clear the cache."""
        with self.lock:
            self.cache.clear()
            self.last_accessed.clear()


class InputPreprocessor:
    """Utility for preprocessing inputs to optimize proof generation."""
    
    @staticmethod
    def hash_input(input_data: Dict[str, Any]) -> str:
        """Create a hash of the input data for caching."""
        input_str = str(sorted([(k, str(v)) for k, v in input_data.items()]))
        return hashlib.md5(input_str.encode()).hexdigest()
    
    @staticmethod
    def preprocess(input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Preprocess input data to optimize for proof generation.
        This can include formatting, normalization, etc.
        """
        # Make a deep copy to avoid modifying the original
        processed = dict(input_data)
        
        # Convert any numpy or torch arrays to regular Python lists
        for key, value in processed.items():
            if hasattr(value, 'tolist'):
                processed[key] = value.tolist()
        
        return processed


class HardwareOptimizer:
    """Utilities for optimizing hardware usage."""
    
    @staticmethod
    def setup_gpu_acceleration() -> bool:
        """
        Configure GPU acceleration if available.
        Returns True if GPU acceleration was enabled.
        """
        try:
            # Try to set up Apple Metal (MPS) acceleration
            if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                bt.logging.info("Using Apple Metal GPU acceleration")
                torch.backends.mps.enable_mps()
                return True
            
            # Try to set up CUDA acceleration
            if torch.cuda.is_available():
                device_count = torch.cuda.device_count()
                bt.logging.info(f"Using CUDA GPU acceleration with {device_count} devices")
                torch.cuda.set_device(0)  # Use the first GPU by default
                return True
                
            bt.logging.info("No GPU acceleration available, using CPU")
            return False
        except Exception as e:
            bt.logging.warning(f"Error setting up GPU acceleration: {e}")
            return False
    
    @staticmethod
    def optimize_memory_usage():
        """Optimize memory usage by clearing caches and running garbage collection."""
        # Clear PyTorch cache if available
        if hasattr(torch, 'cuda') and torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Run garbage collection
        gc.collect()
    
    @staticmethod
    def get_optimal_thread_count() -> int:
        """
        Determine the optimal number of threads to use based on system resources.
        """
        try:
            cpu_count = os.cpu_count() or 4
            
            # Leave 1-2 cores for system operations
            if cpu_count > 4:
                return cpu_count - 2
            elif cpu_count > 2:
                return cpu_count - 1
            else:
                return 1
        except:
            # Default to a reasonable number if we can't determine
            return 4
    
    @staticmethod
    def monitor_system_resources() -> Dict[str, float]:
        """
        Monitor system resources and return a dictionary of metrics.
        """
        try:
            # Get memory usage
            memory = psutil.virtual_memory()
            
            # Get CPU usage
            cpu_percent = psutil.cpu_percent(interval=0.1)
            
            # Get GPU memory usage if available
            gpu_memory_used = 0
            gpu_memory_total = 0
            
            if hasattr(torch, 'cuda') and torch.cuda.is_available():
                gpu_memory_used = torch.cuda.memory_allocated() / 1024**3  # GB
                gpu_memory_total = torch.cuda.get_device_properties(0).total_memory / 1024**3  # GB
            
            return {
                "cpu_percent": cpu_percent,
                "memory_percent": memory.percent,
                "memory_used_gb": memory.used / 1024**3,
                "memory_total_gb": memory.total / 1024**3,
                "gpu_memory_used_gb": gpu_memory_used,
                "gpu_memory_total_gb": gpu_memory_total
            }
        except:
            # Return empty dict if we can't monitor
            return {}


class ProofOptimizer:
    """Utilities for optimizing proof generation and handling."""
    
    @staticmethod
    def optimize_proof_size(proof: str) -> str:
        """
        Attempt to optimize the proof size without compromising integrity.
        This is a placeholder - actual implementation would depend on the
        specific proof format and compression algorithms available.
        """
        # Currently a placeholder - actual implementation would depend on
        # the specific proof format and what optimizations are valid
        return proof
    
    @staticmethod
    def parallelize_proof_generation(generate_fn, input_data, num_threads=None):
        """
        Parallelize proof generation if the function supports it.
        
        Args:
            generate_fn: The function that generates the proof
            input_data: The input data for proof generation
            num_threads: Number of threads to use, or None for auto-determine
            
        Returns:
            The result from the generate_fn
        """
        # This is a simplified implementation
        # A more advanced implementation would split the work into parallel tasks
        if num_threads is None:
            num_threads = HardwareOptimizer.get_optimal_thread_count()
            
        # The actual implementation would depend on how the proof generation can be parallelized
        # For now, we just call the function directly
        return generate_fn(input_data)


# Global cache instance
proof_cache = ProofCache()

# Initialize hardware optimization on import
gpu_enabled = HardwareOptimizer.setup_gpu_acceleration()

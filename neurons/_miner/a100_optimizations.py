"""
Optimizations specifically for NVIDIA A100 GPUs running on Linux Ubuntu.

This module contains specialized optimizations that leverage the architecture
and capabilities of NVIDIA A100 GPUs for maximum performance when generating
zero-knowledge proofs.
"""

import os
import time
import gc
import hashlib
import threading
from typing import Dict, Any, Optional, List, Tuple

import bittensor as bt
import torch
import math

from _miner.optimizations import ProofCache, HardwareOptimizer

# A100-specific constants
A100_MEMORY_GB = 80  # A100 has up to 80GB memory depending on model
A100_TENSOR_CORES = 432  # A100 has 432 Tensor Cores
A100_CUDA_CORES = 6912  # A100 has 6912 CUDA cores
A100_MAX_POWER_WATTS = 400  # Maximum power consumption of A100

# Memory allocation constants for A100
A100_MEMORY_RESERVE_PCT = 0.05  # Reserve 5% for system operations
A100_CACHE_MEMORY_PCT = 0.30  # Use up to 30% for cache
A100_PROOF_GEN_MEMORY_PCT = 0.60  # Use up to 60% for proof generation
A100_OTHER_MEMORY_PCT = 0.05  # Reserve 5% for other operations

# CUDA configuration constants
A100_OPTIMAL_THREADS_PER_BLOCK = 1024  # Optimal threads per block for A100
A100_OPTIMAL_STREAMS = 16  # Optimal number of CUDA streams

# Enhanced cache size for A100's larger memory
A100_MAX_CACHE_SIZE = 2000  # Larger cache size for A100


class A100Config:
    """
    Configuration settings specific to the NVIDIA A100 GPU.
    """
    # GPU detection and configuration
    DEVICE_NAME = "A100"
    MIN_COMPUTE_CAPABILITY = (8, 0)  # A100 has compute capability 8.0
    
    # CUDA optimization settings
    CUDA_CACHE_SIZE_MB = 2048  # 2GB CUDA cache
    JIT_ENABLE = True
    
    # Memory optimization
    MEMORY_GROWTH = True  # Allow memory growth
    MEMORY_RESERVE_PCT = A100_MEMORY_RESERVE_PCT
    
    # Performance settings
    MAX_BATCH_SIZE = 32
    OPTIMAL_TENSOR_PARALLEL = 4
    OPTIMAL_PIPELINE_PARALLEL = 2
    OPTIMAL_STREAMS = A100_OPTIMAL_STREAMS
    
    # Power and thermal management
    POWER_LIMIT_WATTS = 300  # Limiting to 300W for sustained operations
    TARGET_TEMP_CELSIUS = 75  # Target temperature
    MAX_TEMP_CELSIUS = 85  # Maximum temperature
    
    # A100-specific kernel configurations
    THREADS_PER_BLOCK = A100_OPTIMAL_THREADS_PER_BLOCK
    SHARED_MEMORY_SIZE_KB = 48  # A100 has 48KB shared memory per block


class A100ProofCache(ProofCache):
    """
    Enhanced proof cache designed for the A100's larger memory capacity.
    """
    def __init__(self, max_size: int = A100_MAX_CACHE_SIZE):
        super().__init__(max_size=max_size)
        
        # A100-specific enhancements
        self.memory_limit = 0  # Will be set during initialize()
        self.gpu_tensors: Dict[str, torch.Tensor] = {}  # Store tensors directly on GPU
        self.gpu_lock = threading.RLock()
        self.initialized = False
    
    def initialize(self):
        """
        Initialize the GPU cache with A100-specific settings.
        """
        if not torch.cuda.is_available():
            bt.logging.warning("CUDA not available, A100 enhanced cache disabled")
            return False
            
        try:
            # Check if the GPU is an A100
            device_name = torch.cuda.get_device_name(0)
            if A100Config.DEVICE_NAME not in device_name:
                bt.logging.warning(f"GPU is not an A100 (found: {device_name}), some optimizations may not be optimal")
            
            # Set up memory limits
            total_memory = torch.cuda.get_device_properties(0).total_memory
            self.memory_limit = int(total_memory * A100_CACHE_MEMORY_PCT)
            
            bt.logging.info(f"A100 enhanced cache initialized with {self.memory_limit / 1024**3:.2f}GB memory limit")
            self.initialized = True
            return True
            
        except Exception as e:
            bt.logging.error(f"Error initializing A100 enhanced cache: {e}")
            return False
    
    def get(self, key: str) -> Optional[Dict[str, Any]]:
        """
        Get a proof from the cache with A100 enhancements.
        """
        # Use standard cache if not initialized
        if not self.initialized:
            return super().get(key)
        
        with self.lock:
            if key in self.cache:
                self.last_accessed[key] = time.time()
                
                # Check if we have GPU tensor
                if key in self.gpu_tensors:
                    with self.gpu_lock:
                        # Convert GPU tensor back to CPU and update cache
                        gpu_tensor = self.gpu_tensors[key]
                        value = self.cache[key]
                        
                        # Update with latest tensor if needed
                        if 'tensor_data' in value:
                            if hasattr(gpu_tensor, 'to'):
                                value['tensor_data'] = gpu_tensor.to('cpu')
                        
                        return value
                else:
                    # Regular cache hit
                    return self.cache[key]
            
            return None
    
    def set(self, key: str, value: Dict[str, Any]) -> None:
        """
        Add a proof to the cache with A100 enhancements.
        """
        # Use standard cache if not initialized
        if not self.initialized:
            super().set(key, value)
            return
        
        with self.lock:
            # Check if we need to perform cleanup
            current_time = time.time()
            if current_time - self.last_cleanup > 3600:  # 1 hour
                self._cleanup()
            
            # If cache is full, remove least recently used items
            if len(self.cache) >= self.max_size:
                self._evict_lru()
            
            # Store value in standard cache
            self.cache[key] = value
            self.last_accessed[key] = current_time
            
            # Check if any tensor data should be stored on GPU
            tensor_data = None
            if 'tensor_data' in value and hasattr(value['tensor_data'], 'to'):
                tensor_data = value['tensor_data']
            elif 'output' in value and isinstance(value['output'], dict) and 'proof' in value['output']:
                # Try to extract tensor data from proof if available
                proof_data = value['output']['proof']
                if hasattr(proof_data, 'to'):
                    tensor_data = proof_data
            
            # Store tensor on GPU if available
            if tensor_data is not None:
                with self.gpu_lock:
                    # Check current GPU memory usage
                    current_usage = torch.cuda.memory_allocated()
                    if current_usage < self.memory_limit:
                        try:
                            # Move tensor to GPU
                            self.gpu_tensors[key] = tensor_data.to('cuda')
                        except Exception as e:
                            bt.logging.debug(f"Could not move tensor to GPU: {e}")
    
    def clear(self) -> None:
        """
        Clear the cache including GPU tensors.
        """
        with self.lock:
            # Clear standard cache
            self.cache.clear()
            self.last_accessed.clear()
            
            # Clear GPU tensors if initialized
            if self.initialized:
                with self.gpu_lock:
                    for tensor in self.gpu_tensors.values():
                        del tensor
                    self.gpu_tensors.clear()
                    
                    # Force CUDA garbage collection
                    torch.cuda.empty_cache()


class A100Optimizer:
    """
    Optimizer specifically designed for A100 GPU to maximize performance.
    """
    def __init__(self):
        self.device = None
        self.streams = []
        self.is_a100 = False
        self.initialized = False
        self.a100_cache = A100ProofCache()
        
        # Performance tracking
        self.perf_stats = {
            "gpu_utilization": [],
            "memory_utilization": [],
            "temperature": [],
            "power_usage": [],
            "proof_times": []
        }
    
    def initialize(self) -> bool:
        """
        Initialize the A100 optimizations.
        """
        if not torch.cuda.is_available():
            bt.logging.warning("CUDA not available, A100 optimizations disabled")
            return False
            
        try:
            # Get device properties
            device_count = torch.cuda.device_count()
            if device_count == 0:
                bt.logging.warning("No CUDA devices found")
                return False
                
            # Use first device
            self.device = torch.device('cuda:0')
            
            # Check if it's an A100
            device_name = torch.cuda.get_device_name(0)
            self.is_a100 = A100Config.DEVICE_NAME in device_name
            
            if not self.is_a100:
                bt.logging.warning(f"GPU is not an A100 (found: {device_name}), some optimizations may not be optimal")
            
            # Check compute capability
            cc_major, cc_minor = torch.cuda.get_device_capability(0)
            min_major, min_minor = A100Config.MIN_COMPUTE_CAPABILITY
            
            if (cc_major, cc_minor) < (min_major, min_minor):
                bt.logging.warning(f"GPU compute capability ({cc_major}.{cc_minor}) is less than A100 minimum ({min_major}.{min_minor})")
            
            # Configure CUDA settings
            self._configure_cuda()
            
            # Initialize streams
            self._initialize_streams()
            
            # Initialize enhanced cache
            self.a100_cache.initialize()
            
            bt.logging.success(f"A100 optimizations initialized for {device_name}")
            self.initialized = True
            return True
            
        except Exception as e:
            bt.logging.error(f"Error initializing A100 optimizations: {e}")
            return False
    
    def _configure_cuda(self):
        """
        Configure CUDA settings for optimal A100 performance.
        """
        try:
            # Set CUDA cache size
            os.environ['CUDA_CACHE_MAXSIZE'] = str(A100Config.CUDA_CACHE_SIZE_MB)
            
            # Enable JIT compilation
            if A100Config.JIT_ENABLE and hasattr(torch, 'jit'):
                torch._C._jit_set_profiling_executor(True)
                torch._C._jit_set_profiling_mode(True)
            
            # Set tensor math precision
            if hasattr(torch.backends, 'cuda') and hasattr(torch.backends.cuda, 'matmul'):
                torch.backends.cuda.matmul.allow_tf32 = True
                
            if hasattr(torch.backends, 'cudnn'):
                torch.backends.cudnn.benchmark = True
                torch.backends.cudnn.deterministic = False
                torch.backends.cudnn.enabled = True
                if hasattr(torch.backends.cudnn, 'allow_tf32'):
                    torch.backends.cudnn.allow_tf32 = True
            
            # Attempt to set power limit (requires nvidia-smi and root access)
            try:
                import subprocess
                subprocess.run([
                    'nvidia-smi', 
                    '--power-limit={}'.format(A100Config.POWER_LIMIT_WATTS),
                    '-i', '0'
                ], check=False)
            except:
                pass  # Ignore if we can't set power limit
                
        except Exception as e:
            bt.logging.warning(f"Error configuring CUDA settings: {e}")
    
    def _initialize_streams(self):
        """
        Initialize CUDA streams for parallel execution.
        """
        try:
            # Clear existing streams
            self.streams = []
            
            # Create streams
            for _ in range(A100Config.OPTIMAL_STREAMS):
                self.streams.append(torch.cuda.Stream())
                
            bt.logging.debug(f"Initialized {len(self.streams)} CUDA streams")
            
        except Exception as e:
            bt.logging.warning(f"Error initializing CUDA streams: {e}")
    
    def get_stream(self, index=None):
        """
        Get a CUDA stream for parallel execution.
        
        Args:
            index: Optional index of the stream to get. If None, returns the next available stream.
        """
        if not self.initialized or not self.streams:
            return torch.cuda.current_stream()
            
        if index is None:
            # Round-robin selection
            index = int(time.time() * 1000) % len(self.streams)
            
        return self.streams[index % len(self.streams)]
    
    def optimize_memory(self):
        """
        Optimize memory usage for A100 GPUs.
        """
        if not self.initialized:
            return
            
        try:
            # Empty CUDA cache
            torch.cuda.empty_cache()
            
            # Run garbage collection
            gc.collect()
            
            # Get current memory stats
            allocated = torch.cuda.memory_allocated() / 1024**3  # GB
            reserved = torch.cuda.memory_reserved() / 1024**3  # GB
            
            bt.logging.debug(f"A100 memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")
            
        except Exception as e:
            bt.logging.warning(f"Error optimizing A100 memory: {e}")
    
    def prepare_tensors(self, data, stream_index=None):
        """
        Prepare tensors for A100 GPU processing.
        
        This efficiently moves data to the GPU, using pinned memory and streams
        for optimal transfer performance.
        """
        if not self.initialized:
            return data
        
        try:
            stream = self.get_stream(stream_index)
            
            with torch.cuda.stream(stream):
                # Handle different types of data
                if isinstance(data, torch.Tensor):
                    # Pin memory if not already on CUDA
                    if not data.is_cuda:
                        data = data.pin_memory().to(self.device, non_blocking=True)
                    return data
                    
                elif isinstance(data, dict):
                    # Process dictionaries recursively
                    result = {}
                    for k, v in data.items():
                        result[k] = self.prepare_tensors(v, stream_index)
                    return result
                    
                elif isinstance(data, (list, tuple)):
                    # Process lists recursively
                    return type(data)([self.prepare_tensors(x, stream_index) for x in data])
                    
                else:
                    # Return other types unchanged
                    return data
        except Exception as e:
            bt.logging.warning(f"Error preparing tensors for A100: {e}")
            return data
    
    def optimize_for_inference(self, model):
        """
        Optimize a PyTorch model for inference on A100 GPUs.
        """
        if not self.initialized:
            return model
            
        try:
            # Move to GPU
            if hasattr(model, 'to') and callable(model.to):
                model = model.to(self.device)
            
            # Try to use TorchScript for optimization
            if A100Config.JIT_ENABLE and hasattr(torch, 'jit') and hasattr(model, 'eval'):
                try:
                    model.eval()
                    model = torch.jit.script(model)
                    bt.logging.info("Model optimized with TorchScript")
                except Exception as e:
                    bt.logging.debug(f"Could not apply TorchScript optimization: {e}")
            
            return model
            
        except Exception as e:
            bt.logging.warning(f"Error optimizing model for A100: {e}")
            return model
    
    def get_optimal_batch_size(self, model_memory_usage, input_size=1):
        """
        Calculate the optimal batch size based on GPU memory and model requirements.
        
        Args:
            model_memory_usage: Estimated memory usage per sample in bytes
            input_size: Size of a single input sample
        
        Returns:
            The optimal batch size for processing
        """
        if not self.initialized:
            return A100Config.MAX_BATCH_SIZE
            
        try:
            # Get available memory
            total_memory = torch.cuda.get_device_properties(0).total_memory
            allocated_memory = torch.cuda.memory_allocated()
            available_memory = total_memory - allocated_memory
            
            # Reserve some memory for operations
            available_memory *= (1 - A100_MEMORY_RESERVE_PCT)
            
            # Calculate max possible batch size
            max_batch_size = int(available_memory / (model_memory_usage * input_size))
            
            # Ensure batch size is at least 1 and doesn't exceed the config maximum
            optimal_batch_size = max(1, min(max_batch_size, A100Config.MAX_BATCH_SIZE))
            
            # Round to power of 2 for better performance
            optimal_batch_size = 2 ** int(math.log2(optimal_batch_size))
            
            return optimal_batch_size
            
        except Exception as e:
            bt.logging.warning(f"Error calculating optimal batch size: {e}")
            return A100Config.MAX_BATCH_SIZE
    
    def record_performance(self, proof_time=None):
        """
        Record performance metrics for A100 GPU.
        """
        if not self.initialized:
            return
            
        try:
            # Get GPU stats
            utilization = 0
            memory_percent = 0
            temperature = 0
            power = 0
            
            try:
                import subprocess
                import json
                
                result = subprocess.run(
                    ['nvidia-smi', '--query-gpu=utilization.gpu,utilization.memory,temperature.gpu,power.draw', '--format=csv,noheader,nounits'],
                    capture_output=True,
                    text=True,
                    check=True
                )
                
                if result.stdout:
                    values = [float(x.strip()) for x in result.stdout.split(',')]
                    if len(values) >= 4:
                        utilization, memory_percent, temperature, power = values
            except:
                pass  # Ignore if nvidia-smi isn't available
            
            # Record stats
            self.perf_stats["gpu_utilization"].append(utilization)
            self.perf_stats["memory_utilization"].append(memory_percent)
            self.perf_stats["temperature"].append(temperature)
            self.perf_stats["power_usage"].append(power)
            
            if proof_time is not None:
                self.perf_stats["proof_times"].append(proof_time)
                
            # Limit size of stats history
            max_history = 100
            for key in self.perf_stats:
                if len(self.perf_stats[key]) > max_history:
                    self.perf_stats[key] = self.perf_stats[key][-max_history:]
                    
        except Exception as e:
            bt.logging.debug(f"Error recording A100 performance: {e}")
    
    def get_performance_stats(self):
        """
        Get summary of performance statistics.
        """
        if not self.perf_stats["proof_times"]:
            return {}
            
        try:
            # Calculate statistics
            avg_gpu_util = sum(self.perf_stats["gpu_utilization"]) / max(1, len(self.perf_stats["gpu_utilization"]))
            avg_mem_util = sum(self.perf_stats["memory_utilization"]) / max(1, len(self.perf_stats["memory_utilization"]))
            avg_temp = sum(self.perf_stats["temperature"]) / max(1, len(self.perf_stats["temperature"]))
            avg_power = sum(self.perf_stats["power_usage"]) / max(1, len(self.perf_stats["power_usage"]))
            
            # Calculate proof times
            proof_times = self.perf_stats["proof_times"]
            avg_proof_time = sum(proof_times) / len(proof_times)
            min_proof_time = min(proof_times)
            max_proof_time = max(proof_times)
            
            return {
                "avg_gpu_utilization": avg_gpu_util,
                "avg_memory_utilization": avg_mem_util,
                "avg_temperature": avg_temp,
                "avg_power_usage": avg_power,
                "avg_proof_time": avg_proof_time,
                "min_proof_time": min_proof_time,
                "max_proof_time": max_proof_time,
                "proof_count": len(proof_times)
            }
            
        except Exception as e:
            bt.logging.warning(f"Error calculating performance stats: {e}")
            return {}

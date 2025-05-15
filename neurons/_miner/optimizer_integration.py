"""
Integration module for Omron miner optimizations.

This module connects all optimization components and provides a simple API
to apply optimizations to the MinerSession class.
"""

import sys
import os
import traceback
import importlib.util
import bittensor as bt

def apply_optimizations():
    """
    Apply all optimizations to the MinerSession class.
    This function should be called after importing MinerSession but before using it.
    """
    try:
        bt.logging.info("Applying performance optimizations to miner...")
        
        # Import required modules
        from _miner.optimizations import proof_cache, InputPreprocessor, HardwareOptimizer, ProofOptimizer
        from _miner.preload_common_circuits import preload_common_circuits
        from _miner.optimized_pow_handler import optimized_handle_pow_request
        
        # Get MinerSession class
        from _miner.miner_session import MinerSession
        
        # Add preload_common_circuits method to MinerSession
        MinerSession.preload_common_circuits = preload_common_circuits
        
        # Replace handle_pow_request with optimized version
        MinerSession.handle_pow_request = optimized_handle_pow_request
        
        # Add hook to initialize optimization components in __init__
        original_init = MinerSession.__init__
        
        def optimized_init(self):
            # Call original __init__
            original_init(self)
            
            # Import optimizations
            from _miner.optimizations import proof_cache, InputPreprocessor, HardwareOptimizer, ProofOptimizer
            
            # Initialize optimization components
            self.proof_cache = proof_cache  # Use global proof cache
            self.input_preprocessor = InputPreprocessor()
            self.hardware_optimizer = HardwareOptimizer()
            self.proof_optimizer = ProofOptimizer()
            
            # Set optimal thread count
            self.optimal_threads = self.hardware_optimizer.get_optimal_thread_count()
            bt.logging.info(f"Using {self.optimal_threads} threads for optimized processing")
            
            # Setup GPU acceleration if available
            gpu_available = self.hardware_optimizer.setup_gpu_acceleration()
            if gpu_available:
                bt.logging.success("GPU acceleration enabled for improved performance")
            
            # Pre-allocate memory for common operations
            self.preload_common_circuits()
            
            # Increase socket timeout for large proof handling
            import websocket
            websocket.setdefaulttimeout(60)  # Extended timeout for better reliability
            
            bt.logging.success("Miner performance optimizations applied successfully")
        
        # Replace __init__ with optimized version
        MinerSession.__init__ = optimized_init
        
        # Also enhance queryZkProof to use caching for regular proof generation
        original_query_zk = MinerSession.queryZkProof
        
        def optimized_query_zk(self, synapse):
            """Optimized version of queryZkProof using caching and hardware optimization"""
            from _validator.models.request_type import RequestType
            import hashlib
            import time
            
            time_in = time.time()
            
            # Extract model_id and input
            try:
                model_id = synapse.query_input.get("model_id", "")
                input_data = synapse.query_input.get("public_inputs", {})
                
                # Create cache key
                input_hash = hashlib.md5(str(input_data).encode()).hexdigest()
                cache_key = f"zk:{model_id}:{input_hash}"
                
                # Check cache
                cached_result = self.proof_cache.get(cache_key)
                if cached_result:
                    bt.logging.info("📦 Using cached ZK proof result")
                    synapse.query_output = cached_result.get("output", "")
                    time_out = time.time()
                    
                    # Log metrics
                    self.log_batch.append({
                        str(model_id): {
                            "proof_time": cached_result.get("proof_time", 0),
                            "overhead_time": 0.01,
                            "total_response_time": time_out - time_in,
                            "cache_hit": True
                        }
                    })
                    
                    return synapse
                
                # Continue with original implementation but optimize where possible
                result = original_query_zk(self, synapse)
                
                # Cache the result if successful
                if synapse.query_output and "error" not in synapse.query_output.lower():
                    self.proof_cache.set(cache_key, {
                        "output": synapse.query_output,
                        "proof_time": getattr(self, '_last_proof_time', 0),
                        "timestamp": time.time()
                    })
                
                # Optimize memory after intensive operation
                if hasattr(self, 'hardware_optimizer'):
                    self.hardware_optimizer.optimize_memory_usage()
                
                return result
                
            except Exception as e:
                bt.logging.error(f"Error in optimized queryZkProof: {e}")
                traceback.print_exc()
                return original_query_zk(self, synapse)
        
        # Replace queryZkProof with optimized version
        MinerSession.queryZkProof = optimized_query_zk
        
        bt.logging.success("All miner optimizations applied successfully")
        return True
        
    except Exception as e:
        bt.logging.error(f"Failed to apply optimizations: {e}")
        traceback.print_exc()
        return False
        
        
# Apply optimizations automatically when this module is imported
success = apply_optimizations()

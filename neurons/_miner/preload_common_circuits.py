"""
Function to preload common circuits for optimized miner performance.
To be imported in MinerSession.
"""

import traceback
import bittensor as bt

from deployment_layer.circuit_store import circuit_store
from _validator.models.request_type import RequestType

def preload_common_circuits(self):
    """
    Preload commonly used circuits to improve response time.
    """
    try:
        # Get available circuits
        circuits = list(circuit_store.circuits.values())
        if not circuits:
            bt.logging.warning("No circuits available for preloading")
            return
            
        bt.logging.info(f"Preloading {len(circuits)} circuits for improved performance")
        
        # Identify which circuits are most commonly used
        # For now, we'll preload all available circuits, but in a production system,
        # you might want to only preload the most frequently used ones
        
        # Create a small sample input for each circuit type
        for circuit in circuits:
            try:
                # Create a dummy request to prepare the model
                circuit_id = circuit.id
                bt.logging.debug(f"Preloading circuit: {circuit_id}")
                
                # Prepare a simple test input
                # This doesn't actually generate a proof, but loads the circuit into memory
                _ = circuit.input_handler(RequestType.BENCHMARK)
                
                # If the circuit has a model, ensure it's loaded
                if hasattr(circuit, 'model') and circuit.model is not None:
                    # Touch the model to ensure it's loaded
                    _ = circuit.model
                    
                bt.logging.debug(f"Successfully preloaded circuit: {circuit_id}")
            except Exception as e:
                bt.logging.warning(f"Error preloading circuit {circuit.id}: {e}")
                continue
                
        bt.logging.success("Circuit preloading completed")
        
        # Optimize memory after preloading
        if hasattr(self, 'hardware_optimizer'):
            self.hardware_optimizer.optimize_memory_usage()
        
    except Exception as e:
        bt.logging.error(f"Error during circuit preloading: {e}")
        traceback.print_exc()

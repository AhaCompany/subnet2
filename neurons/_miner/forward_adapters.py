"""
Bittensor 9.4.0 adapter module.

This module provides adapters for making forward functions compatible with Bittensor 9.4.0
by handling the strict type checking in the axon.attach method.
"""

import importlib
import inspect
import sys
import traceback
from typing import Any, Callable, Dict, Tuple, Type

import bittensor as bt


def create_synapse_adapter(forward_fn: Callable, synapse_class_name: str) -> Callable:
    """
    Create an adapter for a forward function that accepts synapse objects.
    
    This adapter creates a wrapper function that Bittensor 9.4.0 will recognize
    as having a signature with a parameter inheriting from bt.Synapse.
    
    Args:
        forward_fn: The original forward function to adapt
        synapse_class_name: The name of the synapse class to use ("QueryZkProof", 
                           "ProofOfWeightsSynapse", or "Competition")
        
    Returns:
        A wrapper function that is compatible with axon.attach
    """
    # Create a class that inherits from bt.Synapse with the necessary attributes
    adapter_class_name = f"{synapse_class_name}Adapter"
    adapter_class = type(
        adapter_class_name, 
        (bt.Synapse,), 
        {
            "__module__": __name__,
            "__doc__": f"Adapter for {synapse_class_name} for Bittensor 9.4.0 compatibility"
        }
    )
    
    # Create the adapter function
    def adapter_fn(self, synapse: adapter_class) -> bt.Synapse:
        """
        Adapter function that passes the synapse to the original function.
        Bittensor will see this function as accepting a bt.Synapse subclass.
        """
        try:
            # Pass the original synapse to the original function
            result = forward_fn(self, synapse)
            return result
        except Exception as e:
            bt.logging.error(f"Error in {adapter_class_name} adapter: {e}")
            bt.logging.error(traceback.format_exc())
            # Make sure to return the original synapse in case of error
            return synapse
    
    # Set the correct annotations to make bittensor happy
    adapter_fn.__annotations__ = {"synapse": adapter_class, "return": bt.Synapse}
    adapter_fn.__name__ = f"{forward_fn.__name__}_adapter"
    adapter_fn.__qualname__ = f"{forward_fn.__qualname__}_adapter"
    adapter_fn.__module__ = forward_fn.__module__
    
    return adapter_fn


def attach_with_adapter(
    axon: bt.axon,
    forward_fn: Callable,
    blacklist_fn: Callable,
    synapse_type: str
) -> bool:
    """
    Attach a forward function to an axon using an adapter to make it compatible
    with Bittensor 9.4.0's type checking.
    
    Args:
        axon: The Bittensor axon to attach to
        forward_fn: The original forward function
        blacklist_fn: The blacklist function to use
        synapse_type: The type of synapse ("QueryZkProof", "ProofOfWeightsSynapse", or "Competition")
        
    Returns:
        True if successful, False otherwise
    """
    try:
        adapter_fn = create_synapse_adapter(forward_fn, synapse_type)
        axon.attach(forward_fn=adapter_fn, blacklist_fn=blacklist_fn)
        bt.logging.info(f"Successfully attached {forward_fn.__name__} via adapter")
        return True
    except Exception as e:
        bt.logging.error(f"Failed to attach {forward_fn.__name__} via adapter: {e}")
        bt.logging.error(traceback.format_exc())
        return False
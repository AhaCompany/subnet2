"""
Synapse registry for Bittensor 9.4.0 compatibility.

This module registers custom synapse classes with Bittensor to make them available
for deserialization when requests are received.
"""

import traceback
from typing import Dict, Type, Optional

import bittensor as bt

from protocol import (
    QueryZkProof,
    ProofOfWeightsSynapse,
    Competition,
    QueryForProofAggregation
)

# Keep track of registered synapse classes
registered_synapses: Dict[str, Type] = {}


def register_protocol_synapses():
    """
    Register all protocol synapse classes with Bittensor.
    
    In Bittensor 9.4.0, synapses need to be registered in order to be
    deserialized properly when requests are received.
    """
    global registered_synapses
    
    try:
        # Get the synapse registry from Bittensor
        if hasattr(bt, 'synapse') and hasattr(bt.synapse, 'synapse_registry'):
            registry = bt.synapse.synapse_registry
        else:
            bt.logging.warning("Could not find Bittensor synapse registry")
            return False
        
        # Register each protocol synapse
        synapse_classes = [
            QueryZkProof,
            ProofOfWeightsSynapse,
            Competition,
            QueryForProofAggregation
        ]
        
        for synapse_class in synapse_classes:
            class_name = synapse_class.__name__
            try:
                # Register the synapse with Bittensor
                registry.register_synapse_class(synapse_class)
                registered_synapses[class_name] = synapse_class
                bt.logging.info(f"Registered synapse class: {class_name}")
            except Exception as e:
                bt.logging.error(f"Error registering synapse class {class_name}: {e}")
        
        # Check if registration was successful
        registered_names = list(registry.synapse_classes.keys())
        bt.logging.info(f"Registered synapse classes: {registered_names}")
        
        return True
            
    except Exception as e:
        bt.logging.error(f"Error registering protocol synapses: {e}")
        bt.logging.error(traceback.format_exc())
        return False


def get_registered_synapse(name: str) -> Optional[Type]:
    """
    Get a registered synapse class by name.
    """
    return registered_synapses.get(name)


# Automatically register synapses when this module is imported
register_protocol_synapses()
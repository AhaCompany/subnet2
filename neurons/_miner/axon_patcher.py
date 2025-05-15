"""
Axon patcher module.

This module provides a direct patch to the bittensor.axon.attach method
to bypass the type checking for Synapse class inheritance.
"""

import inspect
import traceback
import types
from typing import Any, Callable, Dict, Tuple, Type, List, Union

import bittensor as bt


def patch_axon_attach():
    """
    Patch the bittensor.axon.attach method to bypass the type checking.
    This is a direct monkey patch to the method to allow attaching any forward_fn.
    
    This is a workaround for compatibility issues with bittensor 9.4.0.
    """
    bt.logging.warning("Patching bittensor.axon.attach method to bypass type checking")
    
    # Store the original attach method
    original_attach = bt.axon.attach
    
    # Create a new attach method that bypasses the type checking
    def patched_attach(self, forward_fn: Callable, blacklist_fn: Callable = None):
        """
        Patched version of attach that bypasses the type checking.
        """
        try:
            # Get the signature of the forward_fn
            sig = inspect.signature(forward_fn)
            params = list(sig.parameters.values())
            
            # Make sure the function has at least two parameters (self + synapse)
            if len(params) < 2:
                raise ValueError("forward_fn must have at least two parameters (self + synapse)")
            
            # Bypass the type checking by skipping the issubclass check
            # Add the endpoint to the axon's forward_fns and blacklist_fns
            self.forward_fns.append(forward_fn)
            if blacklist_fn is not None:
                self.blacklist_fns.append(blacklist_fn)
            else:
                # Add a default blacklist function that always returns False
                self.blacklist_fns.append(lambda synapse: (False, "No blacklist function provided"))
                
            bt.logging.info(f"Successfully attached {forward_fn.__name__} with patched method")
            return True
            
        except Exception as e:
            bt.logging.error(f"Error in patched attach: {e}")
            bt.logging.error(traceback.format_exc())
            return False
    
    # Replace the original attach method with our patched version
    bt.axon.attach = types.MethodType(patched_attach, bt.axon)
    return True


def apply_axon_patch_for_session(miner_session):
    """
    Apply the axon patch and attach all required forward functions for a MinerSession.
    
    Args:
        miner_session: The MinerSession instance to configure
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Patch the axon attach method
        patch_axon_attach()
        
        # Access the axon from the session
        axon = miner_session.axon
        if axon is None:
            bt.logging.error("Axon is not initialized in MinerSession")
            return False
            
        # Attach the forward functions
        bt.axon.attach(axon, forward_fn=miner_session.queryZkProof, blacklist_fn=miner_session.proof_blacklist)
        bt.axon.attach(axon, forward_fn=miner_session.handle_pow_request, blacklist_fn=miner_session.pow_blacklist)
        bt.axon.attach(axon, forward_fn=miner_session.handleCompetitionRequest, blacklist_fn=miner_session.competition_blacklist)
        
        bt.logging.success("Successfully attached all forward functions to axon with patched method")
        return True
        
    except Exception as e:
        bt.logging.error(f"Error applying axon patch: {e}")
        bt.logging.error(traceback.format_exc())
        return False
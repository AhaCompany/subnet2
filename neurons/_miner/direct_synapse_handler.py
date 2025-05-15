"""
Direct Synapse Handler for Bittensor 9.4.0 compatibility.

This module provides a direct handler for Bittensor axon requests without relying on
the Synapse registry or serialization/deserialization mechanisms.
"""

import json
import inspect
import traceback
from typing import Any, Dict, Callable, List, Tuple, Type

import bittensor as bt

# Import base protocol classes for constructing objects
from protocol import QueryZkProof, ProofOfWeightsSynapse, Competition


def monkey_patch_axon():
    """
    Monkey patch Bittensor's axon to use our custom forwarding logic.
    This bypasses Bittensor's Synapse registry and serialization mechanism.
    """
    try:
        # Get the original axon class
        original_axon_class = bt.axon.__class__

        # Store the original process_request method
        original_process = bt.axon.process_request
        
        # Create a new process_request method that handles our protocol's synapse types
        def custom_process_request(self, request_proto: Any) -> Any:
            """
            Custom process_request method that handles requests without using 
            Bittensor's Synapse registry or serialization.
            """
            try:
                # Extract request information
                if not hasattr(request_proto, 'synapse_type') or not hasattr(request_proto, 'synapse_data'):
                    # Not a standard Bittensor request, pass to original handler
                    return original_process(self, request_proto)
                
                synapse_type = request_proto.synapse_type
                synapse_data = request_proto.synapse_data
                
                # Log request for debugging
                bt.logging.debug(f"Received request of type: {synapse_type}")
                bt.logging.debug(f"Request data: {synapse_data[:100]}...")  # Truncate for log
                
                # Handle specific synapse types
                if synapse_type == 'QueryZkProof':
                    return handle_query_zk_proof(self, synapse_data)
                elif synapse_type == 'ProofOfWeightsSynapse':
                    return handle_proof_of_weights(self, synapse_data)
                elif synapse_type == 'Competition':
                    return handle_competition(self, synapse_data)
                else:
                    # Unknown synapse type, pass to original handler
                    bt.logging.warning(f"Unknown synapse type: {synapse_type}, using original handler")
                    return original_process(self, request_proto)
                
            except Exception as e:
                bt.logging.error(f"Error in custom process_request: {e}")
                bt.logging.error(traceback.format_exc())
                # Return an error response
                return {
                    'status_code': 500,
                    'status_message': f"Error processing request: {str(e)}",
                    'result': None
                }
        
        # Replace the original process_request method
        bt.axon.process_request = custom_process_request.__get__(bt.axon, original_axon_class)
        
        bt.logging.info("Successfully patched axon.process_request with custom handler")
        return True
        
    except Exception as e:
        bt.logging.error(f"Error patching axon.process_request: {e}")
        bt.logging.error(traceback.format_exc())
        return False


def handle_query_zk_proof(axon, synapse_data: str) -> Dict[str, Any]:
    """
    Handle a QueryZkProof request by manually constructing the synapse and calling the handler.
    """
    try:
        # Parse synapse data
        data = json.loads(synapse_data)
        
        # Create a QueryZkProof object
        synapse = QueryZkProof()
        synapse.query_input = data.get('query_input', {})
        
        # Find the handler
        handler = None
        for forward_fn in axon.forward_fns:
            if forward_fn.__name__ == 'queryZkProof':
                handler = forward_fn
                break
        
        if not handler:
            raise ValueError("Could not find queryZkProof handler")
        
        # Call the handler
        bt.logging.info(f"Calling queryZkProof handler with input: {synapse.query_input}")
        result = handler(axon, synapse)
        
        # Extract result
        response = {
            'status_code': 200,
            'status_message': "Success",
            'result': {
                'query_output': result.query_output
            }
        }
        
        return response
        
    except Exception as e:
        bt.logging.error(f"Error in handle_query_zk_proof: {e}")
        bt.logging.error(traceback.format_exc())
        return {
            'status_code': 500,
            'status_message': f"Error processing QueryZkProof: {str(e)}",
            'result': None
        }


def handle_proof_of_weights(axon, synapse_data: str) -> Dict[str, Any]:
    """
    Handle a ProofOfWeightsSynapse request by manually constructing the synapse and calling the handler.
    """
    try:
        # Parse synapse data
        data = json.loads(synapse_data)
        
        # Create a ProofOfWeightsSynapse object
        synapse = ProofOfWeightsSynapse()
        synapse.verification_key_hash = data.get('verification_key_hash', '')
        synapse.inputs = data.get('inputs', {})
        synapse.proof_system = data.get('proof_system', None)
        
        # Find the handler
        handler = None
        for forward_fn in axon.forward_fns:
            if forward_fn.__name__ == 'handle_pow_request':
                handler = forward_fn
                break
        
        if not handler:
            raise ValueError("Could not find handle_pow_request handler")
        
        # Call the handler
        bt.logging.info(f"Calling handle_pow_request handler with key hash: {synapse.verification_key_hash}")
        result = handler(axon, synapse)
        
        # Extract result
        response = {
            'status_code': 200,
            'status_message': "Success",
            'result': {
                'proof': result.proof,
                'public_signals': result.public_signals
            }
        }
        
        return response
        
    except Exception as e:
        bt.logging.error(f"Error in handle_proof_of_weights: {e}")
        bt.logging.error(traceback.format_exc())
        return {
            'status_code': 500,
            'status_message': f"Error processing ProofOfWeightsSynapse: {str(e)}",
            'result': None
        }


def handle_competition(axon, synapse_data: str) -> Dict[str, Any]:
    """
    Handle a Competition request by manually constructing the synapse and calling the handler.
    """
    try:
        # Parse synapse data
        data = json.loads(synapse_data)
        
        # Create a Competition object
        synapse = Competition()
        synapse.id = data.get('id', 0)
        synapse.hash = data.get('hash', '')
        synapse.file_name = data.get('file_name', '')
        
        # Find the handler
        handler = None
        for forward_fn in axon.forward_fns:
            if forward_fn.__name__ == 'handleCompetitionRequest':
                handler = forward_fn
                break
        
        if not handler:
            raise ValueError("Could not find handleCompetitionRequest handler")
        
        # Call the handler
        bt.logging.info(f"Calling handleCompetitionRequest handler with id: {synapse.id}, hash: {synapse.hash}")
        result = handler(axon, synapse)
        
        # Extract result
        response = {
            'status_code': 200,
            'status_message': "Success",
            'result': {
                'id': result.id,
                'hash': result.hash,
                'file_name': result.file_name,
                'file_content': result.file_content,
                'commitment': result.commitment,
                'error': result.error
            }
        }
        
        return response
        
    except Exception as e:
        bt.logging.error(f"Error in handle_competition: {e}")
        bt.logging.error(traceback.format_exc())
        return {
            'status_code': 500,
            'status_message': f"Error processing Competition: {str(e)}",
            'result': None
        }


# Initialize handlers when module is imported
def setup_direct_handlers():
    """
    Setup direct handlers for axon requests.
    """
    return monkey_patch_axon()
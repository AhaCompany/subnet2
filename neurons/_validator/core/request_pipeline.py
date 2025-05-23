from __future__ import annotations

import copy
import traceback
import random

import bittensor as bt

from _validator.api import ValidatorAPI
from _validator.config import ValidatorConfig
from _validator.core.request import Request
from _validator.models.request_type import RequestType
from _validator.pow.proof_of_weights_handler import ProofOfWeightsHandler
from _validator.scoring.score_manager import ScoreManager
from _validator.utils.hash_guard import HashGuard
from constants import (
    BATCHED_PROOF_OF_WEIGHTS_MODEL_ID,
    SINGLE_PROOF_OF_WEIGHTS_MODEL_ID,
)
from deployment_layer.circuit_store import circuit_store
from execution_layer.circuit import Circuit, CircuitType
from execution_layer.generic_input import GenericInput
from protocol import ProofOfWeightsSynapse, QueryZkProof
from utils.wandb_logger import safe_log
from execution_layer.base_input import BaseInput


class RequestPipeline:
    def __init__(
        self, config: ValidatorConfig, score_manager: ScoreManager, api: ValidatorAPI
    ):
        self.config = config
        self.score_manager = score_manager
        self.api = api
        self.hash_guard = HashGuard()

    def prepare_requests(self, filtered_uids) -> list[Request]:
        """
        Prepare requests for the current validation step.
        This includes both regular benchmark requests and any external requests.

        Args:
            filtered_uids (list): List of UIDs to send requests to.

        Returns:
            list[Request]: List of prepared requests.
        """
        if len(filtered_uids) == 0:
            bt.logging.error("No UIDs to query")
            return []

        if self.api.external_requests_queue:
            return self._prepare_real_world_requests(filtered_uids)
        return self._prepare_benchmark_requests(filtered_uids)

    def _check_and_create_request(
        self,
        uid: int,
        synapse: ProofOfWeightsSynapse | QueryZkProof,
        circuit: Circuit,
        request_type: RequestType,
        request_hash: str | None = None,
        save: bool = False,
    ) -> Request | None:
        """Check hash and create request if valid."""
        try:
            if isinstance(synapse, ProofOfWeightsSynapse):
                input_data = synapse.inputs
            else:
                input_data = synapse.query_input["public_inputs"]

            self.hash_guard.check_hash(input_data)
        except Exception as e:
            bt.logging.error(f"Hash already exists: {e}")
            safe_log({"hash_guard_error": 1})
            if request_type == RequestType.RWR:
                self.api.set_request_result(
                    request_hash, {"success": False, "error": "Hash already exists"}
                )
            return None

        return Request(
            uid=uid,
            axon=self.config.metagraph.axons[uid],
            synapse=synapse,
            circuit=circuit,
            request_type=request_type,
            inputs=GenericInput(RequestType.RWR, input_data),
            request_hash=request_hash,
            save=save,
        )

    def _prepare_real_world_requests(self, filtered_uids: list[int]) -> list[Request]:
        external_request = self.api.external_requests_queue.pop()
        requests = []

        for uid in filtered_uids:
            try:
                synapse, save = self.get_synapse_request(
                    RequestType.RWR, external_request.circuit, external_request
                )
                request = self._check_and_create_request(
                    uid=uid,
                    synapse=synapse,
                    circuit=external_request.circuit,
                    request_type=RequestType.RWR,
                    request_hash=external_request.hash,
                    save=save,
                )
                if request:
                    requests.append(request)
            except Exception as e:
                bt.logging.error(f"Error preparing request for UID {uid}: {e}")
                traceback.print_exc()
                self.api.set_request_result(
                    external_request.hash,
                    {"success": False, "error": "Error preparing request"},
                )
                continue
        return requests

    def _prepare_benchmark_requests(self, filtered_uids: list[int]) -> list[Request]:
        circuit = self.select_circuit_for_benchmark()
        if circuit is None:
            bt.logging.error("No circuit selected")
            return []

        requests = []
        for uid in filtered_uids:
            synapse, save = self.get_synapse_request(RequestType.BENCHMARK, circuit)
            request = self._check_and_create_request(
                uid=uid,
                synapse=synapse,
                circuit=circuit,
                request_type=RequestType.BENCHMARK,
                save=save,
            )
            if request:
                requests.append(request)
        return requests

    def select_circuit_for_benchmark(self) -> Circuit:
        """
        Select a circuit for benchmarking using weighted random selection.
        """
        circuits = list(circuit_store.circuits.values())

        return random.choices(
            circuits,
            weights=[
                (circuit.metadata.benchmark_choice_weight or 0) for circuit in circuits
            ],
            k=1,
        )[0]

    def format_for_query(
        self, inputs: dict[str, object] | BaseInput, circuit: Circuit
    ) -> dict[str, object]:
        if hasattr(inputs, "to_json"):
            inputs = inputs.to_json()
        return {"public_inputs": inputs, "model_id": circuit.id}

    def get_synapse_request(
        self,
        request_type: RequestType,
        circuit: Circuit,
        request: any | None = None,
    ) -> tuple[ProofOfWeightsSynapse | QueryZkProof, bool]:
        inputs = (
            circuit.input_handler(request_type)
            if request_type == RequestType.BENCHMARK
            else circuit.input_handler(
                RequestType.RWR,
                copy.deepcopy(request.inputs),
            )
        )

        if request_type == RequestType.RWR:
            if circuit.metadata.type == CircuitType.PROOF_OF_WEIGHTS:
                return (
                    ProofOfWeightsSynapse(
                        subnet_uid=circuit.metadata.netuid,
                        verification_key_hash=circuit.id,
                        proof_system=circuit.proof_system,
                        inputs=inputs.to_json(),
                        proof="",
                        public_signals="",
                    ),
                    True,
                )
            return (
                QueryZkProof(
                    model_id=circuit.id,
                    query_input=self.format_for_query(inputs, circuit),
                    query_output="",
                ),
                True,
            )

        if circuit.id in [
            SINGLE_PROOF_OF_WEIGHTS_MODEL_ID,
            BATCHED_PROOF_OF_WEIGHTS_MODEL_ID,
        ]:
            synapse_request, save = ProofOfWeightsHandler.prepare_pow_request(
                circuit, self.score_manager
            )
            if synapse_request:
                return synapse_request, save

        if circuit.metadata.type == CircuitType.PROOF_OF_COMPUTATION:
            return (
                QueryZkProof(
                    model_id=circuit.id,
                    query_input=self.format_for_query(inputs, circuit),
                    query_output="",
                ),
                False,
            )

        return (
            ProofOfWeightsSynapse(
                subnet_uid=circuit.metadata.netuid,
                verification_key_hash=circuit.id,
                proof_system=circuit.proof_system,
                inputs=inputs.to_json(),
                proof="",
                public_signals="",
            ),
            False,
        )

    def prepare_single_request(self, uid: int) -> Request | None:
        """
        Prepare a single request for a specific UID.

        Args:
            uid (int): The UID to prepare a request for.

        Returns:
            Request | None: The prepared request, or None if preparation failed.
        """
        if self.api.external_requests_queue:
            requests = self._prepare_real_world_requests([uid])
        else:
            requests = self._prepare_benchmark_requests([uid])

        return requests[0] if requests else None

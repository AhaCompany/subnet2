import traceback

# isort: off
import cli_parser  # <- this need to stay before bittensor import

import bittensor as bt

# isort: on

# Debug để hiểu vấn đề
print(f"Bittensor version: {bt.__version__}")
import sys
print(f"Python version: {sys.version}")

# Import các module tối ưu hóa sau khi đã import bittensor
from _miner.optimizer_integration import apply_optimizations
from _miner.a100_integration import apply_a100_optimizations

from _miner.miner_session import MinerSession
from constants import Roles
from utils import run_shared_preflight_checks

if __name__ == "__main__":
    cli_parser.init_config(Roles.MINER)
    run_shared_preflight_checks(Roles.MINER)

    try:
        # Initialize the circuit store and load external models
        from deployment_layer.circuit_store import circuit_store

        circuit_store.load_circuits()

        bt.logging.info("Creating miner session...")
        miner_session = MinerSession()
        
        # Áp dụng tối ưu hóa chung cho miner
        bt.logging.info("Applying optimizations...")
        apply_optimizations()
        
        # Áp dụng tối ưu hóa đặc biệt cho A100 GPU
        # bt.logging.info("Applying A100 optimizations...")
        # apply_a100_optimizations(miner_session)
        
        bt.logging.debug("Running main loop...")
        miner_session.run()
    except Exception:
        bt.logging.error(
            f"CRITICAL: Failed to run miner session\n{traceback.format_exc()}"
        )

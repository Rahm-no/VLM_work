"""Stub of vLLM's KV-connector base class, so PallasConnector can run without vLLM (tests only)."""
import sys
import types


def install():
    name = "vllm.distributed.kv_transfer.kv_connector.v1.base"
    if name in sys.modules and hasattr(sys.modules[name], "KVConnectorBase_V1"):
        return
    base = types.ModuleType(name)

    class KVConnectorMetadata:
        pass

    class KVConnectorBase_V1:
        def __init__(self, vllm_config, role, kv_cache_config=None):
            self._kv_transfer_config = vllm_config.kv_transfer_config
            self._connector_metadata = None

        def bind_connector_metadata(self, m):
            self._connector_metadata = m

        def clear_connector_metadata(self):
            self._connector_metadata = None

        def _get_connector_metadata(self):
            return self._connector_metadata

        def has_connector_metadata(self):
            return self._connector_metadata is not None

    base.KVConnectorBase_V1, base.KVConnectorMetadata, base.KVConnectorRole = (
        KVConnectorBase_V1, KVConnectorMetadata, object)
    for parent in ["vllm", "vllm.distributed", "vllm.distributed.kv_transfer",
                   "vllm.distributed.kv_transfer.kv_connector", "vllm.distributed.kv_transfer.kv_connector.v1"]:
        sys.modules.setdefault(parent, types.ModuleType(parent))
    sys.modules[name] = base

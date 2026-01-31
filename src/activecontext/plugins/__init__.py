"""Context Application Protocol (CAP) — extensible node plugin system.

This package provides the plugin infrastructure for defining, registering,
and managing context node types — both local (in-process Python) and remote
(cross-language via JSON-RPC over stdio using the CAP wire protocol).

Architecture:
- protocol.py: NodePlugin protocol that all node types satisfy
- cap_transport.py: CAPTransport abstract protocol for all transports
- wire.py: CAP wire protocol message catalog (JSON-RPC 2.0)
- descriptor.py: Plugin descriptors for registration metadata
- manager.py: Plugin server lifecycle management
- transport.py: StdioTransport (default stdio implementation)
- connection.py: Connection lifecycle and schema handshake
- remote_node.py: Proxy node for remote RPC plugins
"""

from activecontext.plugins.cap_transport import CAPTransport
from activecontext.plugins.descriptor import NodePluginDescriptor, PluginSource
from activecontext.plugins.protocol import NodePlugin
from activecontext.plugins.transport import StdioTransport

__all__ = [
    "CAPTransport",
    "NodePlugin",
    "NodePluginDescriptor",
    "PluginSource",
    "StdioTransport",
]

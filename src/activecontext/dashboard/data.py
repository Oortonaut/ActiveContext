"""Data collection functions for dashboard API responses."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from activecontext.core.llm.discovery import (
    get_available_models,
    get_available_providers,
)

if TYPE_CHECKING:
    from activecontext.session.session_manager import Session


def get_llm_status(current_model: str | None) -> dict[str, Any]:
    """Get LLM provider and model status."""
    providers = get_available_providers()
    models = get_available_models()

    return {
        "current_model": current_model,
        "available_providers": providers,
        "available_models": [
            {
                "model_id": m.model_id,
                "name": m.name,
                "provider": m.provider,
                "description": m.description,
            }
            for m in models
        ],
        "configured": len(providers) > 0,
    }


def get_session_summary(
    session: Session,
    model: str | None = None,
    mode: str | None = None,
) -> dict[str, Any]:
    """Get summary info about a session."""
    return {
        "session_id": session.session_id,
        "cwd": str(session.cwd),
        "model": model or (session.llm.model if session.llm else None),
        "mode": mode,
    }


def get_context_data(session: Session) -> dict[str, Any]:
    """Get all context objects organized by type with hierarchy info.
    
    Returns all node types dynamically - no filtering.
    """
    try:
        graph = session.get_context_graph()
    except Exception:
        return {"nodes_by_type": {}, "total": 0}

    nodes_by_type: dict[str, list[dict[str, Any]]] = {}
    total = 0

    for node in graph:
        total += 1
        try:
            if hasattr(node, "GetDigest"):
                digest = node.GetDigest()
                obj_type = digest.get("type", "unknown")

                # Add parent/children info for hierarchy visualization
                if hasattr(node, "parent_ids"):
                    digest["parent_ids"] = list(node.parent_ids)
                else:
                    digest["parent_ids"] = []
                if hasattr(node, "children_ids"):
                    digest["children_ids"] = list(node.children_ids)
                else:
                    digest["children_ids"] = []

                # Group by type dynamically
                if obj_type not in nodes_by_type:
                    nodes_by_type[obj_type] = []
                nodes_by_type[obj_type].append(digest)
        except Exception:
            # Skip nodes that fail to serialize
            continue

    return {"nodes_by_type": nodes_by_type, "total": total}


def get_timeline_data(session: Session) -> dict[str, Any]:
    """Get statement execution timeline."""
    try:
        timeline = session.timeline
        statements = timeline.get_statements()

        # Access private _executions dict directly (no public API)
        executions_dict: dict[str, list[Any]] = getattr(timeline, "_executions", {})

        statement_list: list[dict[str, Any]] = []
        for stmt in statements:
            try:
                # Get execution results for this statement
                executions = executions_dict.get(stmt.statement_id, [])
                latest = executions[-1] if executions else None

                statement_list.append({
                    "statement_id": stmt.statement_id,
                    "index": stmt.index,
                    "source": stmt.source,
                    "timestamp": stmt.timestamp,
                    "status": latest.status.value if latest else "pending",
                    "duration_ms": latest.duration_ms if latest else 0,
                    "has_error": latest.exception is not None if latest else False,
                })
            except Exception:
                continue

        return {"statements": statement_list, "count": len(statement_list)}
    except Exception:
        return {"statements": [], "count": 0}


def get_projection_data(session: Session) -> dict[str, Any]:
    """Get token budget and projection section breakdown."""
    # Get config for total_budget (used in fallback too)
    try:
        from activecontext.config import get_config

        config = get_config()
        total_budget = config.projection.total_budget or 8000
    except Exception:
        total_budget = 8000

    try:
        projection = session.get_projection()

        sections: list[dict[str, Any]] = []
        total_used = 0

        for section in projection.sections:
            try:
                sections.append({
                    "type": section.section_type,
                    "source_id": section.source_id,
                    "tokens_used": section.tokens_used,
                    "state": section.state.name.lower() if section.state else "details",
                })
                total_used += section.tokens_used
            except Exception:
                continue

        return {
            "total_budget": projection.token_budget,
            "total_used": total_used,
            "utilization": total_used / projection.token_budget if projection.token_budget else 0,
            "sections": sections,
        }
    except Exception:
        return {
            "total_budget": total_budget,
            "total_used": 0,
            "utilization": 0,
            "sections": [],
        }


def get_message_history_data(session: Session) -> dict[str, Any]:
    """Get the exact message history for display.
    
    Returns the full conversation history from session._message_history,
    which contains the exact messages for replay and debugging.
    This is separate from the rendered conversation sent to the LLM.
    """
    messages: list[dict[str, Any]] = []
    
    try:
        conversation = getattr(session, "_message_history", [])
        for i, msg in enumerate(conversation):
            try:
                # Message is typically a dict with role and content
                if isinstance(msg, dict):
                    msg_data = {
                        "id": f"msg_{i}",
                        "role": msg.get("role", "unknown"),
                        "content": msg.get("content", ""),
                        "originator": msg.get("originator"),
                        "tool_name": msg.get("tool_name"),
                        "tool_args": msg.get("tool_args"),
                    }
                else:
                    # Message object from activecontext.core.llm.provider
                    # Role is an Enum, so access .value for the string
                    role = getattr(msg, "role", None)
                    if hasattr(role, "value"):
                        role = role.value
                    msg_data = {
                        "id": f"msg_{i}",
                        "role": role or "unknown",
                        "content": getattr(msg, "content", ""),
                        "originator": getattr(msg, "originator", None),
                        "tool_name": getattr(msg, "tool_name", None),
                        "tool_args": getattr(msg, "tool_args", None),
                    }
                messages.append(msg_data)
            except Exception:
                continue
    except Exception:
        pass
    
    return {"messages": messages, "count": len(messages)}


def get_rendered_projection_data(session: Session) -> dict[str, Any]:
    """Get the full rendered projection content sent to the LLM.
    
    This shows exactly what the LLM sees in its context window.
    """
    try:
        projection = session.get_projection()
        
        # Get full rendered content
        rendered = projection.render()
        
        # Also get per-section breakdown for detailed view
        sections: list[dict[str, Any]] = []
        for section in projection.sections:
            try:
                sections.append({
                    "type": section.section_type,
                    "source_id": section.source_id,
                    "content": section.content,
                    "tokens_used": section.tokens_used,
                    "state": section.state.name.lower() if section.state else "details",
                })
            except Exception:
                continue
        
        # Count total tokens
        from activecontext.core.tokens import count_tokens
        total_tokens = count_tokens(rendered)
        
        return {
            "rendered": rendered,
            "total_tokens": total_tokens,
            "token_budget": projection.token_budget,
            "sections": sections,
            "section_count": len(sections),
        }
    except Exception as e:
        return {
            "rendered": f"Error rendering projection: {e}",
            "total_tokens": 0,
            "token_budget": 0,
            "sections": [],
            "section_count": 0,
        }


def get_client_capabilities_data() -> dict[str, Any]:
    """Get ACP client capabilities and transport info for dashboard."""
    from activecontext.dashboard.server import (
        get_client_info,
        get_protocol_version,
        get_transport_type,
    )

    transport_type = get_transport_type()
    client_info = get_client_info()
    protocol_version = get_protocol_version()

    return {
        "transport": {
            "type": transport_type,
            "is_acp": transport_type == "acp",
        },
        "protocol_version": protocol_version,
        "client": client_info,
    }


def get_session_features_data(
    session: Session,
    model: str | None = None,
    mode: str | None = None,
) -> dict[str, Any]:
    """Get session features including mode, model, working directory, and client capabilities."""
    from activecontext.dashboard.server import (
        get_client_info,
        get_protocol_version,
        get_transport_type,
    )

    # Get client/transport info
    transport_type = get_transport_type()
    client_info = get_client_info()
    protocol_version = get_protocol_version()

    return {
        "model": model or (session.llm.model if session.llm else None),
        "mode": mode,
        "cwd": str(session.cwd),
        "title": getattr(session, "title", None),
        # Transport and client info
        "transport": {
            "type": transport_type,
            "is_acp": transport_type == "acp",
        },
        "protocol_version": protocol_version,
        "client": client_info,
    }


def format_session_update(
    kind: str,
    session_id: str,
    payload: dict[str, Any],
    timestamp: float,
) -> dict[str, Any]:
    """Format a session update for WebSocket broadcast."""
    return {
        "type": "update",
        "kind": kind,
        "session_id": session_id,
        "timestamp": timestamp,
        "payload": payload,
    }

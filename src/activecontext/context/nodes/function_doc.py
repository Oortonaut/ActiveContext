"""FunctionDocNode - Function signature and docstring extraction."""

from __future__ import annotations

import inspect
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from activecontext.context.graph import LinkedChildOrder
from activecontext.context.state import Expansion, TickFrequency
from activecontext.context.traceable import trace_all_fields

from .base import ContextNode


@trace_all_fields
@dataclass(kw_only=True)
class FunctionDocNode(ContextNode):
    """Extract and display function signatures and docstrings.

    Attributes:
        file_path: Path to Python file
        function_name: Name of function to document
        signature: Extracted function signature
        docstring: Extracted docstring
        source_lines: Optional full source code
    """

    file_path: str = ""
    function_name: str = ""
    signature: str = ""
    docstring: str = ""
    source_lines: str = ""

    def GetDigest(self) -> dict[str, Any]:
        return {
            "id": self.node_id,
            "type": self.node_type,
            "file_path": self.file_path,
            "function_name": self.function_name,
            "has_docstring": bool(self.docstring),
            "expansion": self.default_expansion.value,
        }

    def extract_function_info(self) -> FunctionDocNode:
        """Extract function signature and docstring from file."""
        try:
            import ast

            with open(self.file_path, encoding="utf-8") as f:
                source = f.read()

            tree = ast.parse(source, filename=self.file_path)

            # Find the function
            for node in ast.walk(tree):
                # Type guard: narrow node to FunctionDef or AsyncFunctionDef
                if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    continue
                if node.name == self.function_name:
                    # Extract signature
                    args = []
                    for arg in node.args.args:
                        arg_str = arg.arg
                        if arg.annotation:
                            arg_str += f": {ast.unparse(arg.annotation)}"
                        args.append(arg_str)

                    returns = ""
                    if node.returns:
                        returns = f" -> {ast.unparse(node.returns)}"

                    async_prefix = "async " if isinstance(node, ast.AsyncFunctionDef) else ""
                    arg_list = ", ".join(args)
                    self.signature = f"{async_prefix}def {node.name}({arg_list}){returns}"

                    # Extract docstring
                    self.docstring = ast.get_docstring(node) or ""

                    # Extract source
                    self.source_lines = ast.unparse(node)

                    self.mark_changed(f"Extracted {self.function_name}")
                    return self

            self.docstring = f"[Function {self.function_name} not found in {self.file_path}]"
        except Exception as e:
            self.docstring = f"[Error extracting function: {e}]"

        return self

    def render_content(self) -> str:
        """Render content: signature + docstring."""
        if not self.signature:
            self.extract_function_info()

        parts = [f"{self.signature}\n"]
        if self.docstring:
            parts.append(f'"""\n{self.docstring}\n"""\n')
        return "".join(parts)

    def render_digest(self) -> str:
        """Return function doc indicator."""
        return f"DOC: {self.function_name}"

    def to_dict(self) -> dict[str, Any]:
        """Serialize FunctionDocNode to dict."""
        data = super().to_dict()
        data.update(
            {
                "file_path": self.file_path,
                "function_name": self.function_name,
                "signature": self.signature,
                "docstring": self.docstring,
                "source_lines": self.source_lines,
            }
        )
        return data

    @classmethod
    def _from_dict(cls, data: dict[str, Any]) -> FunctionDocNode:
        """Deserialize FunctionDocNode from dict."""
        tick_freq = None
        if data.get("tick_frequency"):
            tick_freq = TickFrequency.from_dict(data["tick_frequency"])

        return cls(
            node_id=data["node_id"],
            parent_ids=set(data.get("parent_ids", [])),
            child_order=LinkedChildOrder.from_list(
                data.get("child_order") or data.get("children_ids") or []
            ),
            default_expansion=Expansion(data.get("expansion", "all")),
            mode=data.get("mode", "paused"),
            tick_frequency=tick_freq,
            version=data.get("version", 0),
            created_at=data.get("created_at", time.time()),
            updated_at=data.get("updated_at", time.time()),
            display_sequence=data.get("display_sequence"),
            originator=data.get("originator"),
            title=data.get("title", ""),
            file_path=data.get("file_path", ""),
            function_name=data.get("function_name", ""),
            signature=data.get("signature", ""),
            docstring=data.get("docstring", ""),
            source_lines=data.get("source_lines", ""),
        )

    @classmethod
    def from_method(
        cls,
        method: Callable[..., Any],
        *,
        default_expansion: Expansion = Expansion.HEADER,
    ) -> FunctionDocNode:
        """Create a FunctionDocNode from a method object.

        Extracts the signature and docstring directly from the method
        using introspection, without needing to read from a file.

        Args:
            method: The method or function to document.
            default_expansion: Default expansion level for the node.

        Returns:
            FunctionDocNode with signature and docstring populated.
        """
        # Get function name
        function_name = getattr(method, "__name__", str(method))

        # Get signature
        try:
            sig = inspect.signature(method)
            # Check if method is async
            is_async = inspect.iscoroutinefunction(method)
            async_prefix = "async " if is_async else ""
            signature = f"{async_prefix}def {function_name}{sig}"
        except (ValueError, TypeError):
            signature = f"def {function_name}(...)"

        # Get docstring
        docstring = inspect.getdoc(method) or ""

        # Get source file path if available
        try:
            file_path = inspect.getfile(method)
        except (TypeError, OSError):
            file_path = ""

        return cls(
            function_name=function_name,
            signature=signature,
            docstring=docstring,
            file_path=file_path,
            default_expansion=default_expansion,
            title=function_name,
            tracing=False,
        )

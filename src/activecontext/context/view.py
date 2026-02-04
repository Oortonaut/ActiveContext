"""View layer for context nodes.

NodeView provides view-specific state (hidden, expansion) while delegating
content operations to the underlying ContextNode.

Architecture:
- Content graph: DAG of ContextNodes for ticking and token rollup
- View graph: Flat list of NodeViews for rendering order

NodeView owns visibility state, ContextNode owns content data.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from activecontext.context import trace_all_fields
from activecontext.context.state import Expansion

if TYPE_CHECKING:
    from activecontext.context.nodes import ContextNode
    from activecontext.context.state import NotificationLevel


@trace_all_fields
@dataclass(init=False)
class NodeView:
    """View wrapper for ContextNode with visibility state.

    Separates view concerns (hidden, expansion) from content concerns (data, ticking).
    The DSL binds variables to NodeViews, allowing view-specific state
    while forwarding content operations to the underlying node.

    Attributes:
        node: The underlying ContextNode (content)
        hidden: Whether this view is hidden from projection
        expansion: Expansion state for rendering (HEADER, CONTENT, INDEX, ALL)
    """

    node: ContextNode
    hidden: bool
    expansion: Expansion
    indent: int = 0

    def __init__(
        self,
        node: ContextNode,
        indent: int = 0,
        hidden: bool | None = None,
        expansion: Expansion | None = None,
    ) -> None:
        """Create a view wrapping a node.

        Args:
            node: The ContextNode to wrap
            hidden: Whether the view is hidden (default: node.default_hidden)
            expansion: Expansion state (default: node.default_expansion)
        """
        hidden = node.default_hidden if hidden is None else hidden
        expansion = node.default_expansion if expansion is None else expansion

        object.__setattr__(self, "node", node)
        object.__setattr__(self, "hidden", hidden)
        object.__setattr__(self, "expansion", expansion)
        object.__setattr__(self, "indent", indent)

    # --- Rendering ---

    def render_header(self) -> str:
        """Render uniform header line with token counts at current expansion."""
        from activecontext.context.state import NotificationLevel

        node = self.node
        token_info = node.get_token_breakdown()
        token_str = token_info.format_token_info(self.expansion)

        # Build display_id: "text_1" or fallback to node_id
        seq = node.display_sequence
        display_id = f"{node.node_type}_{seq}" if seq is not None else node.node_id

        name = node.render_digest()

        brief = self.expansion.value
        nl = node.notification_level
        if nl and nl != NotificationLevel.IGNORE:
            brief = f"{brief} {nl.value}"

        return f"{name} | {{#{display_id}}} {brief} {token_str}\n"

    def render(self, text_buffers: dict[str, Any] | None = None) -> str:
        """Render this view at current expansion level."""
        if self.expansion == Expansion.HEADER:
            return self.render_header()
        header = self.render_header()
        content = self.node.render_content(text_buffers=text_buffers)
        return header + content

    # --- Token Calculations ---

    @property
    def visible_tokens(self) -> int:
        """Tokens visible at current expansion level.

        Maps directly to Expansion enum:
        - HEADER: header_tokens
        - CONTENT: header + content
        - INDEX: header + content + index
        - ALL: all_tokens (header + content + children recursively)

        Returns:
            0 if hidden, otherwise tokens based on expansion state.
        """
        if self.hidden:
            return 0

        node = self.node
        if self.expansion == Expansion.HEADER:
            return node.header_tokens
        elif self.expansion == Expansion.CONTENT:
            return node.header_tokens + node.content_tokens
        elif self.expansion == Expansion.INDEX:
            return node.header_tokens + node.content_tokens + node.index_tokens
        else:  # ALL
            return node.all_tokens

    def Run(self, freq: Any = None) -> NodeView:
        """Enable tick recomputation with given frequency."""
        self.node.Run(freq)
        return self

    def Pause(self) -> NodeView:
        """Disable tick recomputation."""
        self.node.Pause()
        return self

    def SetNotify(self, level: NotificationLevel) -> NodeView:
        """Set notification level."""
        self.node.notification_level = level
        return self

    # --- Attribute Forwarding ---

    def __getattr__(self, name: str) -> Any:
        """Forward attribute access to the underlying node."""
        return getattr(self.node, name)

    def __setattr__(self, name: str, value: Any) -> None:
        """Set attributes with type checking for view fields.

        View fields (hidden, expansion, node) are stored locally with type validation.
        Other attributes are forwarded to the underlying node.
        """
        if name == "expansion":
            if not isinstance(value, Expansion):
                raise TypeError(f"expansion must be Expansion, got {type(value).__name__}")
            object.__setattr__(self, name, value)
        elif name == "hidden":
            if not isinstance(value, bool):
                raise TypeError(f"hidden must be bool, got {type(value).__name__}")
            object.__setattr__(self, name, value)
        elif name == "node":
            object.__setattr__(self, name, value)
        else:
            setattr(self.node, name, value)

    def to_dict(self) -> dict[str, Any]:
        """Serialize view state for session persistence."""
        return {
            "type": "NodeView",
            "node_id": self.node.node_id,
            "hidden": self.hidden,
            "expansion": self.expansion.value,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any], node: ContextNode) -> NodeView:
        """Restore view from serialized state."""
        return cls(
            node,
            hidden=data.get("hidden", False),
            expansion=Expansion(data["expansion"]) if "expansion" in data else None,
        )

    def __repr__(self) -> str:
        """Return string representation."""
        hidden_str = ", hidden=True" if self.hidden else ""
        return f"NodeView({self.node!r}, expansion={self.expansion.value}{hidden_str})"

    def __eq__(self, other: Any) -> bool:
        """Compare views by their underlying node."""
        if isinstance(other, NodeView):
            return self.node is other.node
        if isinstance(other, type(self.node)):
            return self.node is other
        return False

    def __hash__(self) -> int:
        """Hash by underlying node ID."""
        return hash(self.node.node_id)


class ChoiceView(NodeView):
    """View that provides dropdown-like selection behavior for nodes with children.

    ChoiceView is a view-layer concern - any node can become a "choice" by wrapping
    it in ChoiceView. Selection controls which children are visible.

    Behavior by expansion mode:
    - HEADER/CONTENT/INDEX: No changes to children via apply_selection
    - INDEX: Use render_index() to get header lines for all children
    - ALL: Only selected child visible (with its own expansion mode),
           or all children hidden if no selection

    Rendering helpers:
    - render_index(): Header line for each child (for INDEX mode)
    - render_digest(): "selected [A | B | C]" format

    Example:
        # Wrap any node in ChoiceView
        choice = ChoiceView(group_node, selected_id="child-b")

        # Select different option (fluent API)
        choice.select("option-2")

        # Get INDEX data (children headers)
        print(choice.render_index())

    Attributes:
        _selected_id: ID of the currently selected child (or None)
    """

    _selected_id: str | None

    def __init__(
        self,
        node: ContextNode,
        selected_id: str | None = None,
        hidden: bool | None = None,
        expansion: Expansion | None = None,
    ) -> None:
        """Create a choice view wrapping a node.

        Args:
            node: The ContextNode to wrap
            selected_id: ID of the initially selected child (or None)
            hidden: Whether the view is hidden (default: node.default_hidden)
            expansion: Expansion state (default: node.default_expansion)
        """
        super().__init__(node, hidden=hidden, expansion=expansion)
        object.__setattr__(self, "_selected_id", selected_id)

    @property
    def selected_id(self) -> str | None:
        """ID of the currently selected child."""
        return self._selected_id

    @selected_id.setter
    def selected_id(self, value: str | None) -> None:
        """Set the selected child ID."""
        object.__setattr__(self, "_selected_id", value)

    def select(self, child_id: str) -> ChoiceView:
        """Select a child by ID (fluent API).

        Args:
            child_id: ID of the child to select

        Returns:
            Self for method chaining
        """
        object.__setattr__(self, "_selected_id", child_id)
        return self

    def __setattr__(self, name: str, value: Any) -> None:
        """Handle attribute assignment, including _selected_id."""
        if name == "_selected_id":
            object.__setattr__(self, name, value)
        elif name == "selected_id":
            # Property setter - use descriptor protocol
            prop = type(self).__dict__.get(name)
            if prop is not None and hasattr(prop, "__set__"):
                prop.__set__(self, value)
            else:
                object.__setattr__(self, name, value)
        else:
            super().__setattr__(name, value)

    def _get_child_ids(self) -> list[str]:
        """Get ordered list of child IDs from the node."""
        return self.node.child_order.to_list()

    def apply_selection(self, views: dict[str, NodeView]) -> None:
        """Apply selection filtering to child views based on expansion mode.

        Behavior by expansion mode:
        - HEADER/CONTENT/INDEX: No changes to children (INDEX renders headers itself)
        - ALL: Only selected child visible (with its own expansion mode),
               or all children hidden if no selection

        Args:
            views: Dict mapping node_id -> NodeView for all views
        """
        if self.expansion != Expansion.ALL:
            return  # Only filter in ALL mode; INDEX renders headers itself

        child_ids = self._get_child_ids()

        # ALL mode: show only selected child, hide all if no selection
        for child_id in child_ids:
            if child_id in views:
                if self._selected_id is None:
                    views[child_id].hidden = True
                else:
                    views[child_id].hidden = child_id != self._selected_id
                # Selected child keeps its own expansion mode

    def get_options(self) -> list[str]:
        """Get titles of all child options.

        Returns:
            List of child titles in child_order order
        """
        node = self.node
        graph = getattr(node, "_graph", None)
        if graph is None:
            return []

        # Get child IDs in order
        child_ids = node.child_order.to_list()

        # Get titles for each child
        titles = []
        for child_id in child_ids:
            child = graph.get_node(child_id)
            if child:
                title = getattr(child, "title", None) or child_id
                titles.append(title)
        return titles

    def render_index(self) -> str:
        """Render INDEX data: header line for each child option.

        Returns:
            Newline-separated headers of all children
        """
        node = self.node
        graph = getattr(node, "_graph", None)
        if graph is None:
            return ""

        child_ids = self._get_child_ids()
        lines = []
        for child_id in child_ids:
            child = graph.get_node(child_id)
            if child:
                lines.append(child.render_digest())
        return "\n".join(lines)

    def render_digest(self) -> str:
        """Render as 'selected [A | B | C]'.

        Returns:
            Brief string showing selected option and all choices
        """
        options = self.get_options()
        if not options:
            return "[No options]"

        # Get the selected child's title
        selected_title = None
        if self._selected_id:
            node = self.node
            graph = getattr(node, "_graph", None)
            if graph:
                selected_child = graph.get_node(self._selected_id)
                if selected_child:
                    selected_title = getattr(selected_child, "title", None) or self._selected_id

        if selected_title is None:
            selected_title = options[0] if options else "?"

        return f"{selected_title} [{' | '.join(options)}]"

    def to_dict(self) -> dict[str, Any]:
        """Serialize view state for session persistence."""
        return {
            "type": "ChoiceView",
            "node_id": self.node.node_id,
            "hidden": self.hidden,
            "expansion": self.expansion.value,
            "selected_id": self._selected_id,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any], node: ContextNode) -> ChoiceView:
        """Restore view from serialized state."""
        return cls(
            node,
            selected_id=data.get("selected_id"),
            hidden=data.get("hidden", False),
            expansion=Expansion(data["expansion"]) if "expansion" in data else None,
        )

    def __repr__(self) -> str:
        """Return string representation."""
        hidden_str = ", hidden=True" if self.hidden else ""
        selected_str = f", selected={self._selected_id!r}" if self._selected_id else ""
        return (
            f"ChoiceView({self.node!r}, expansion={self.expansion.value}{hidden_str}{selected_str})"
        )


class SequenceView(ChoiceView):
    """View for ordered sequential progression through children.

    SequenceView extends ChoiceView to add progression semantics:
    - Agent works through steps in order
    - Current step is visible, others are hidden (like ChoiceView)
    - Tracks completion state per step
    - Supports forward/backward navigation

    Rendering:
    - Progress header: "## Workflow Progress [2/3]"
    - Completed steps marked with [x], current with [>], pending with [ ]

    Example:
        # Create sequence of review steps
        seq = SequenceView(group_node)

        # Progress through steps
        seq.advance()       # Move to next step
        seq.mark_complete() # Mark current done without advancing
        seq.back()          # Go back one step
        seq.skip()          # Skip current step

        # Check status
        print(seq.progress)      # "2/3"
        print(seq.is_complete)   # True when all steps done
    """

    _current_index: int
    _completed_steps: set[int]

    def __init__(
        self,
        node: ContextNode,
        selected_id: str | None = None,
        hidden: bool | None = None,
        expansion: Expansion | None = None,
    ) -> None:
        """Create a sequence view wrapping a node.

        Args:
            node: The ContextNode to wrap (typically a GroupNode)
            selected_id: Initial selection (default: first child)
            hidden: Whether the view is hidden (default: node.default_hidden)
            expansion: Expansion state (default: ALL for full content)
        """
        # Default expansion to ALL for sequences
        if expansion is None:
            expansion = Expansion.ALL

        super().__init__(node, selected_id=selected_id, hidden=hidden, expansion=expansion)

        # Initialize progression state with defaults
        object.__setattr__(self, "_current_index", 0)
        object.__setattr__(self, "_completed_steps", set())

        # Set initial selection based on index
        child_ids = self._get_child_ids()
        if child_ids and self._selected_id is None and 0 <= self._current_index < len(child_ids):
            object.__setattr__(self, "_selected_id", child_ids[self._current_index])

    def to_dict(self) -> dict[str, Any]:
        """Serialize view state for session persistence."""
        return {
            "type": "SequenceView",
            "node_id": self.node.node_id,
            "hidden": self.hidden,
            "expansion": self.expansion.value,
            "selected_id": self._selected_id,
            "current_index": self._current_index,
            "completed_steps": list(self._completed_steps),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any], node: ContextNode) -> SequenceView:
        """Restore view from serialized state."""
        view = cls(
            node,
            selected_id=data.get("selected_id"),
            hidden=data.get("hidden", False),
            expansion=Expansion(data["expansion"]) if "expansion" in data else None,
        )
        # Restore progression state
        object.__setattr__(view, "_current_index", data.get("current_index", 0))
        object.__setattr__(view, "_completed_steps", set(data.get("completed_steps", [])))
        # Sync selection with restored index
        view._sync_selection()
        return view

    def _sync_selection(self) -> None:
        """Sync selected_id with current_index."""
        child_ids = self._get_child_ids()
        if child_ids and 0 <= self._current_index < len(child_ids):
            object.__setattr__(self, "_selected_id", child_ids[self._current_index])

    @property
    def current_index(self) -> int:
        """Get current step index (0-based)."""
        return self._current_index

    @property
    def completed_steps(self) -> set[int]:
        """Get set of completed step indices."""
        return self._completed_steps.copy()

    @property
    def total_steps(self) -> int:
        """Get total number of steps."""
        return len(self._get_child_ids())

    @property
    def is_complete(self) -> bool:
        """Check if all steps are completed."""
        total = self.total_steps
        if total == 0:
            return True
        return len(self._completed_steps) >= total

    @property
    def progress(self) -> str:
        """Get progress string like '2/3'."""
        return f"{len(self._completed_steps)}/{self.total_steps}"

    def advance(self) -> SequenceView:
        """Mark current step complete and move to next (fluent API).

        Returns:
            Self for method chaining
        """
        # Mark current as complete
        self._completed_steps.add(self._current_index)

        # Move to next step
        child_ids = self._get_child_ids()
        if self._current_index < len(child_ids) - 1:
            object.__setattr__(self, "_current_index", self._current_index + 1)
            self._sync_selection()

        return self

    def back(self) -> SequenceView:
        """Move to previous step (fluent API).

        Does not change completion status of any step.

        Returns:
            Self for method chaining
        """
        if self._current_index > 0:
            object.__setattr__(self, "_current_index", self._current_index - 1)
            self._sync_selection()
        return self

    def mark_complete(self) -> SequenceView:
        """Mark current step as complete without advancing (fluent API).

        Returns:
            Self for method chaining
        """
        self._completed_steps.add(self._current_index)
        return self

    def skip(self) -> SequenceView:
        """Skip current step without marking complete (fluent API).

        Returns:
            Self for method chaining
        """
        child_ids = self._get_child_ids()
        if self._current_index < len(child_ids) - 1:
            object.__setattr__(self, "_current_index", self._current_index + 1)
            self._sync_selection()
        return self

    def goto(self, index: int) -> SequenceView:
        """Jump to a specific step index (fluent API).

        Args:
            index: Step index to go to (0-based)

        Returns:
            Self for method chaining
        """
        child_ids = self._get_child_ids()
        if 0 <= index < len(child_ids):
            object.__setattr__(self, "_current_index", index)
            self._sync_selection()
        return self

    def render_progress(self) -> str:
        """Render progress list showing all steps with status.

        Returns:
            Markdown formatted progress list:
            - [x] Step 1: Complete
            - [>] Step 2: Current
            - [ ] Step 3: Pending
        """
        child_ids = self._get_child_ids()
        node = self.node
        graph = getattr(node, "_graph", None)

        lines = [f"## Workflow Progress [{self.progress}]"]

        for i, child_id in enumerate(child_ids):
            # Determine status marker
            if i in self._completed_steps:
                marker = "[x]"
            elif i == self._current_index:
                marker = "[>]"
            else:
                marker = "[ ]"

            # Get child title
            title = child_id
            if graph:
                child = graph.get_node(child_id)
                if child:
                    title = getattr(child, "title", None) or child_id

            # Add current marker
            current_marker = " \u2190 current" if i == self._current_index else ""
            lines.append(f"- {marker} Step {i + 1}: {title}{current_marker}")

        return "\n".join(lines)

    def __setattr__(self, name: str, value: Any) -> None:
        """Handle attribute assignment for sequence-specific fields."""
        if name in ("_current_index", "_completed_steps"):
            object.__setattr__(self, name, value)
        else:
            super().__setattr__(name, value)

    def __repr__(self) -> str:
        """Return string representation."""
        hidden_str = ", hidden=True" if self.hidden else ""
        return f"SequenceView({self.node!r}, progress={self.progress}{hidden_str})"


class LoopView(NodeView):
    """View for iterative refinement loops.

    LoopView wraps a single child node and tracks iteration state:
    - Counts iterations
    - Accumulates state across iterations
    - Supports early exit via done()
    - Optional max_iterations limit

    Rendering:
    - Header shows iteration count: "## Review Loop [iteration 2/5]"
    - State dictionary shown before content

    Example:
        # Create iterative review loop
        loop = LoopView(review_node, max_iterations=5)

        # Iterate with feedback
        loop.iterate(feedback="Add error handling")
        loop.iterate(feedback="Improve naming", approved=False)
        loop.iterate(feedback="Looks good!", approved=True)
        loop.done()  # Exit loop early

        # Check status
        print(loop.iteration)     # 3
        print(loop.state)         # {'feedback': 'Looks good!', 'approved': True}
        print(loop.is_done)       # True
    """

    _iteration: int
    _state: dict[str, Any]
    _done: bool
    _max_iterations: int | None

    def __init__(
        self,
        node: ContextNode,
        max_iterations: int | None = None,
        hidden: bool | None = None,
        expansion: Expansion | None = None,
    ) -> None:
        """Create a loop view wrapping a node.

        Args:
            node: The ContextNode to wrap
            max_iterations: Maximum iterations allowed (None = unlimited)
            hidden: Whether the view is hidden (default: node.default_hidden)
            expansion: Expansion state (default: ALL)
        """
        if expansion is None:
            expansion = Expansion.ALL

        super().__init__(node, hidden=hidden, expansion=expansion)

        # Initialize loop state with defaults
        object.__setattr__(self, "_iteration", 1)
        object.__setattr__(self, "_state", {})
        object.__setattr__(self, "_done", False)
        object.__setattr__(self, "_max_iterations", max_iterations)

    def to_dict(self) -> dict[str, Any]:
        """Serialize view state for session persistence."""
        return {
            "type": "LoopView",
            "node_id": self.node.node_id,
            "hidden": self.hidden,
            "expansion": self.expansion.value,
            "max_iterations": self._max_iterations,
            "iteration": self._iteration,
            "state": dict(self._state),
            "done": self._done,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any], node: ContextNode) -> LoopView:
        """Restore view from serialized state."""
        view = cls(
            node,
            max_iterations=data.get("max_iterations"),
            hidden=data.get("hidden", False),
            expansion=Expansion(data["expansion"]) if "expansion" in data else None,
        )
        # Restore loop state
        object.__setattr__(view, "_iteration", data.get("iteration", 1))
        object.__setattr__(view, "_state", dict(data.get("state", {})))
        object.__setattr__(view, "_done", data.get("done", False))
        return view

    @property
    def iteration(self) -> int:
        """Get current iteration number (1-based)."""
        return self._iteration

    @property
    def state(self) -> dict[str, Any]:
        """Get accumulated state dictionary."""
        return self._state.copy()

    @property
    def max_iterations(self) -> int | None:
        """Get maximum iterations limit."""
        return self._max_iterations

    @property
    def is_done(self) -> bool:
        """Check if loop is complete.

        Loop is complete if:
        - done() was called explicitly
        - max_iterations reached
        """
        if self._done:
            return True
        if self._max_iterations is not None:
            return self._iteration > self._max_iterations
        return False

    @property
    def iterations_remaining(self) -> int | None:
        """Get remaining iterations (None if unlimited)."""
        if self._max_iterations is None:
            return None
        return max(0, self._max_iterations - self._iteration + 1)

    def iterate(self, **state_updates: Any) -> LoopView:
        """Increment iteration and update state (fluent API).

        Args:
            **state_updates: Key-value pairs to merge into state

        Returns:
            Self for method chaining
        """
        if self.is_done:
            return self  # No-op if already done

        # Update state
        self._state.update(state_updates)

        # Increment iteration
        object.__setattr__(self, "_iteration", self._iteration + 1)

        return self

    def update_state(self, **state_updates: Any) -> LoopView:
        """Update state without incrementing iteration (fluent API).

        Args:
            **state_updates: Key-value pairs to merge into state

        Returns:
            Self for method chaining
        """
        self._state.update(state_updates)
        return self

    def done(self) -> LoopView:
        """Mark loop as complete (fluent API).

        Returns:
            Self for method chaining
        """
        object.__setattr__(self, "_done", True)
        return self

    def reset(self) -> LoopView:
        """Reset loop to initial state (fluent API).

        Returns:
            Self for method chaining
        """
        object.__setattr__(self, "_iteration", 1)
        object.__setattr__(self, "_state", {})
        object.__setattr__(self, "_done", False)
        return self

    def render_header(self) -> str:
        """Render loop header with iteration info.

        Returns:
            Header string like "## Review Loop [iteration 2/5]"
        """
        if self._max_iterations:
            iter_str = f"iteration {self._iteration}/{self._max_iterations}"
        else:
            iter_str = f"iteration {self._iteration}"

        title = getattr(self.node, "title", None) or "Loop"
        return f"## {title} [{iter_str}]"

    def render_state(self) -> str:
        """Render accumulated state.

        Returns:
            Markdown formatted state display
        """
        if not self._state:
            return ""

        lines = ["**State:**"]
        for key, value in self._state.items():
            lines.append(f"- {key}: {value!r}")
        return "\n".join(lines)

    def __setattr__(self, name: str, value: Any) -> None:
        """Handle attribute assignment for loop-specific fields."""
        if name in ("_iteration", "_state", "_done", "_max_iterations"):
            object.__setattr__(self, name, value)
        else:
            super().__setattr__(name, value)

    def __repr__(self) -> str:
        """Return string representation."""
        hidden_str = ", hidden=True" if self.hidden else ""
        max_str = f"/{self._max_iterations}" if self._max_iterations else ""
        done_str = ", done" if self._done else ""
        return (
            f"LoopView({self.node!r}, iteration={self._iteration}{max_str}{done_str}{hidden_str})"
        )


class StateView(ChoiceView):
    """View for state machine navigation through named states.

    StateView extends ChoiceView to add state machine semantics:
    - Named states mapped to child nodes
    - Transition rules defining valid state changes
    - Current state visible, others hidden
    - State history tracking

    Rendering:
    - Header shows current state and available transitions
    - "## Task State: working -> [done, idle]"

    Example:
        # Create state machine for task workflow
        fsm = StateView(
            group_node,
            states={"idle": "idle-node", "working": "working-node", "done": "done-node"},
            transitions={
                "idle": ["working"],
                "working": ["done", "idle"],
                "done": []
            },
            initial="idle"
        )

        # Navigate states
        fsm.transition("working")
        print(fsm.can_transition("done"))  # True
        print(fsm.can_transition("idle"))  # True
        fsm.transition("done")

        # Check history
        print(fsm.state_history)  # ["idle", "working"]
    """

    _states: dict[str, str]  # state_name -> node_id
    _transitions: dict[str, list[str]]  # state_name -> allowed next states
    _current_state: str
    _state_history: list[str]

    def __init__(
        self,
        node: ContextNode,
        states: dict[str, str] | None = None,
        transitions: dict[str, list[str]] | None = None,
        initial: str | None = None,
        hidden: bool | None = None,
        expansion: Expansion | None = None,
    ) -> None:
        """Create a state machine view wrapping a node.

        Args:
            node: The ContextNode to wrap (typically a GroupNode)
            states: Mapping of state names to child node IDs
            transitions: Mapping of state names to allowed next states
            initial: Initial state name (default: first state)
            hidden: Whether the view is hidden (default: node.default_hidden)
            expansion: Expansion state (default: ALL)
        """
        if expansion is None:
            expansion = Expansion.ALL

        # Initialize states and transitions
        states = states or {}
        transitions = transitions or {}

        # Default initial state to first key
        if initial is None and states:
            initial = next(iter(states.keys()))

        # Get initial node ID for ChoiceView selection
        initial_node_id = states.get(initial) if initial else None

        super().__init__(node, selected_id=initial_node_id, hidden=hidden, expansion=expansion)

        # Initialize state machine with defaults
        object.__setattr__(self, "_states", states)
        object.__setattr__(self, "_transitions", transitions)
        object.__setattr__(self, "_current_state", initial or "")
        object.__setattr__(self, "_state_history", [])

    def to_dict(self) -> dict[str, Any]:
        """Serialize view state for session persistence."""
        return {
            "type": "StateView",
            "node_id": self.node.node_id,
            "hidden": self.hidden,
            "expansion": self.expansion.value,
            "selected_id": self._selected_id,
            "states": self._states,
            "transitions": self._transitions,
            "current_state": self._current_state,
            "state_history": list(self._state_history),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any], node: ContextNode) -> StateView:
        """Restore view from serialized state."""
        view = cls(
            node,
            states=data.get("states", {}),
            transitions=data.get("transitions", {}),
            initial=data.get("current_state"),  # Use saved current as initial
            hidden=data.get("hidden", False),
            expansion=Expansion(data["expansion"]) if "expansion" in data else None,
        )
        # Restore state history
        object.__setattr__(view, "_state_history", list(data.get("state_history", [])))
        return view

    @property
    def current_state(self) -> str:
        """Get current state name."""
        return self._current_state

    @property
    def state_history(self) -> list[str]:
        """Get list of previous states (not including current)."""
        return self._state_history.copy()

    @property
    def valid_transitions(self) -> list[str]:
        """Get list of valid next states from current state."""
        return self._transitions.get(self._current_state, []).copy()

    @property
    def all_states(self) -> list[str]:
        """Get list of all state names."""
        return list(self._states.keys())

    def can_transition(self, to_state: str) -> bool:
        """Check if transition to given state is allowed.

        Args:
            to_state: Target state name

        Returns:
            True if transition is allowed
        """
        allowed = self._transitions.get(self._current_state, [])
        return to_state in allowed

    def transition(self, to_state: str) -> StateView:
        """Transition to a new state (fluent API).

        Args:
            to_state: Target state name

        Returns:
            Self for method chaining

        Raises:
            ValueError: If transition is not allowed
        """
        if not self.can_transition(to_state):
            allowed = self.valid_transitions
            raise ValueError(
                f"Cannot transition from '{self._current_state}' to '{to_state}'. "
                f"Allowed: {allowed}"
            )

        if to_state not in self._states:
            raise ValueError(f"Unknown state: '{to_state}'")

        # Record history
        if self._current_state:
            self._state_history.append(self._current_state)

        # Update state
        object.__setattr__(self, "_current_state", to_state)

        # Update selection to show new state's node
        object.__setattr__(self, "_selected_id", self._states[to_state])

        return self

    def force_transition(self, to_state: str) -> StateView:
        """Force transition to a state, ignoring transition rules (fluent API).

        Use with caution - this bypasses the state machine rules.

        Args:
            to_state: Target state name

        Returns:
            Self for method chaining
        """
        if to_state not in self._states:
            raise ValueError(f"Unknown state: '{to_state}'")

        if self._current_state:
            self._state_history.append(self._current_state)

        object.__setattr__(self, "_current_state", to_state)
        object.__setattr__(self, "_selected_id", self._states[to_state])

        return self

    def reset(self) -> StateView:
        """Reset to initial state, clearing history (fluent API).

        Returns:
            Self for method chaining
        """
        if self._states:
            initial = next(iter(self._states.keys()))
            object.__setattr__(self, "_current_state", initial)
            object.__setattr__(self, "_selected_id", self._states[initial])
        object.__setattr__(self, "_state_history", [])
        return self

    def render_header(self) -> str:
        """Render state header with current state and transitions.

        Returns:
            Header like "## Task State: working -> [done, idle]"
        """
        title = getattr(self.node, "title", None) or "State"
        transitions = self.valid_transitions
        trans_str = f" \u2192 [{', '.join(transitions)}]" if transitions else " (terminal)"
        return f"## {title}: {self._current_state}{trans_str}"

    def __setattr__(self, name: str, value: Any) -> None:
        """Handle attribute assignment for state-specific fields."""
        if name in ("_states", "_transitions", "_current_state", "_state_history"):
            object.__setattr__(self, name, value)
        else:
            super().__setattr__(name, value)

    def __repr__(self) -> str:
        """Return string representation."""
        hidden_str = ", hidden=True" if self.hidden else ""
        history_len = len(self._state_history)
        history_str = f", history={history_len}" if history_len else ""
        return f"StateView({self.node!r}, state={self._current_state!r}{history_str}{hidden_str})"


# View type registry for deserialization
_VIEW_TYPES: dict[str, type[NodeView]] = {
    "NodeView": NodeView,
    "ChoiceView": ChoiceView,
    "SequenceView": SequenceView,
    "LoopView": LoopView,
    "StateView": StateView,
}


def view_from_dict(data: dict[str, Any], node: ContextNode) -> NodeView:
    """Restore a view from serialized state.

    Factory function that dispatches to the appropriate view class
    based on the "type" field in the serialized data.

    Args:
        data: Serialized view dict with "type" and "node_id" fields.
        node: The ContextNode to wrap.

    Returns:
        Restored NodeView subclass instance.

    Raises:
        ValueError: If the view type is unknown.
    """
    view_type = data.get("type", "NodeView")
    view_class = _VIEW_TYPES.get(view_type)
    if view_class is None:
        raise ValueError(f"Unknown view type: {view_type}")
    return view_class.from_dict(data, node)

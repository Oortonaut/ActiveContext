"""View layer for context nodes.

NodeView provides view-specific state (hide, expand) while delegating
content operations to the underlying ContextNode.

Architecture:
- Content graph: DAG of ContextNodes for ticking and token rollup
- View graph: Flat list of NodeViews for rendering order

NodeView owns visibility state, ContextNode owns content data.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from activecontext.context.state import Expansion

if TYPE_CHECKING:
    from activecontext.context.nodes import ContextNode


class NodeView:
    """View wrapper for ContextNode with visibility state.

    Separates view concerns (hide, expand) from content concerns (data, ticking).
    The DSL binds variables to NodeViews, allowing view-specific state
    while forwarding content operations to the underlying node.

    Attributes:
        _node: The underlying ContextNode (content)
        _hide: Whether this view is hidden from projection
        _expand: Expansion state for rendering (HEADER, CONTENT, INDEX, ALL)
        _tags: View-specific metadata (e.g., _hidden_expansion for unhide restoration)
    """

    __slots__ = ("_node", "_hide", "_expand", "_tags")

    _node: ContextNode
    _hide: bool
    _expand: Expansion
    _tags: dict[str, Any]

    def __init__(
        self,
        node: ContextNode,
        hide: bool = False,
        expand: Expansion | None = None,
    ) -> None:
        """Create a view wrapping a node.

        Args:
            node: The ContextNode to wrap
            hide: Whether the view is hidden (default False)
            expand: Expansion state (default: uses node.expansion for migration)
        """
        object.__setattr__(self, "_node", node)
        object.__setattr__(self, "_hide", hide)
        # Migration: use node.expansion as default if not specified
        object.__setattr__(self, "_expand", expand if expand is not None else node.expansion)
        object.__setattr__(self, "_tags", {})

    def node(self) -> ContextNode:
        """Return the underlying ContextNode."""
        return self._node

    # --- View State Properties ---

    @property
    def hide(self) -> bool:
        """Whether this view is hidden from projection."""
        return self._hide

    @hide.setter
    def hide(self, value: bool) -> None:
        """Set hide state."""
        object.__setattr__(self, "_hide", value)

    @property
    def expand(self) -> Expansion:
        """Expansion state for rendering."""
        return self._expand

    @expand.setter
    def expand(self, value: Expansion) -> None:
        """Set expansion state."""
        object.__setattr__(self, "_expand", value)

    @property
    def tags(self) -> dict[str, Any]:
        """View-specific metadata."""
        return self._tags

    # --- Backward Compatibility: expansion property ---

    @property
    def expansion(self) -> Expansion:
        """Alias for expand (backward compatibility)."""
        return self._expand

    @expansion.setter
    def expansion(self, value: Expansion) -> None:
        """Alias for expand (backward compatibility)."""
        object.__setattr__(self, "_expand", value)

    # --- Token Calculations ---

    @property
    def visible_tokens(self) -> int:
        """Tokens visible at current expand level.

        Returns:
            0 if hidden, otherwise tokens based on expand state.
        """
        if self._hide:
            return 0

        node = self._node
        if self._expand == Expansion.HEADER:
            return node.header_tokens
        elif self._expand == Expansion.CONTENT:
            return node.header_tokens + getattr(node, "summary_tokens", 0)
        else:  # DETAILS
            return node.header_tokens + getattr(node, "content_tokens", node.tokens)

    # --- Fluent API ---

    def SetHide(self, hide: bool) -> NodeView:
        """Set hide state (fluent API).

        Args:
            hide: Whether to hide this view
        """
        self.hide = hide
        return self

    def SetExpand(self, expand: Expansion) -> NodeView:
        """Set expansion state (fluent API).

        Args:
            expand: New expansion state (HEADER, CONTENT, INDEX, ALL)
        """
        self.expand = expand
        return self

    def SetExpansion(self, s: Expansion) -> NodeView:
        """Set expansion state (backward compatibility).

        Args:
            s: New expansion state
        """
        self.expand = s
        # Also update node for backward compatibility during migration
        self._node.expansion = s
        return self

    def SetTokens(self, n: int) -> NodeView:
        """Set token budget (fluent API)."""
        self._node.tokens = n
        return self

    def Run(self, freq: Any = None) -> NodeView:
        """Enable tick recomputation with given frequency."""
        self._node.Run(freq)
        return self

    def Pause(self) -> NodeView:
        """Disable tick recomputation."""
        self._node.Pause()
        return self

    def SetNotify(self, level: Any) -> NodeView:
        """Set notification level."""
        self._node.notification_level = level
        return self

    # --- Attribute Forwarding ---

    def __getattr__(self, name: str) -> Any:
        """Forward attribute access to the underlying node."""
        return getattr(self._node, name)

    def __setattr__(self, name: str, value: Any) -> None:
        """Forward attribute assignment to the underlying node.

        Properties (hide, expand, expansion, tags) go through their setters.
        Slot attributes (_node, _hide, _expand, _tags) use object.__setattr__.
        Other attributes are forwarded to the underlying node.
        """
        # Slot attributes - use object.__setattr__
        if name in ("_node", "_hide", "_expand", "_tags"):
            object.__setattr__(self, name, value)
        # Property setters - let descriptor protocol handle it
        elif name in ("hide", "expand", "expansion"):
            # Get the property descriptor from the class and call its setter
            prop = type(self).__dict__.get(name)
            if prop is not None and hasattr(prop, "__set__"):
                prop.__set__(self, value)
            else:
                object.__setattr__(self, name, value)
        else:
            setattr(self._node, name, value)

    def __repr__(self) -> str:
        """Return string representation."""
        hide_str = ", hide=True" if self._hide else ""
        return f"NodeView({self._node!r}, expand={self._expand.value}{hide_str})"

    def __eq__(self, other: Any) -> bool:
        """Compare views by their underlying node."""
        if isinstance(other, NodeView):
            return self._node is other._node
        if isinstance(other, type(self._node)):
            return self._node is other
        return False

    def __hash__(self) -> int:
        """Hash by underlying node ID."""
        return hash(self._node.node_id)


class ChoiceView(NodeView):
    """View that provides dropdown-like selection behavior for nodes with children.

    ChoiceView is a view-layer concern - any node can become a "choice" by wrapping
    it in ChoiceView. Selection controls which children are visible.

    Behavior by expansion mode:
    - HEADER/CONTENT/INDEX: No changes to children via apply_selection
    - INDEX: Use render_index() to get header lines for all children
    - ALL: Only selected child visible (with its own expand mode),
           or all children hidden if no selection

    Rendering helpers:
    - render_index(): Header line for each child (for INDEX mode)
    - render_brief(): "selected [A | B | C]" format

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

    __slots__ = ("_selected_id",)

    _selected_id: str | None

    def __init__(
        self,
        node: ContextNode,
        selected_id: str | None = None,
        hide: bool = False,
        expand: Expansion | None = None,
    ) -> None:
        """Create a choice view wrapping a node.

        Args:
            node: The ContextNode to wrap
            selected_id: ID of the initially selected child (or None)
            hide: Whether the view is hidden (default False)
            expand: Expansion state (default: uses node.expansion)
        """
        super().__init__(node, hide=hide, expand=expand)
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
        """Handle attribute assignment, including _selected_id slot."""
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
        node = self._node
        child_order = getattr(node, "child_order", None)
        if child_order is not None:
            # child_order might be a LinkedChildOrder with to_list() or a regular list
            if hasattr(child_order, "to_list"):
                return child_order.to_list()
            else:
                return list(child_order)
        else:
            return list(getattr(node, "children_ids", set()))

    def apply_selection(self, views: dict[str, NodeView]) -> None:
        """Apply selection filtering to child views based on expansion mode.

        Behavior by expansion mode:
        - HEADER/CONTENT/INDEX: No changes to children (INDEX renders headers itself)
        - ALL: Only selected child visible (with its own expand mode),
               or all children hidden if no selection

        Args:
            views: Dict mapping node_id -> NodeView for all views
        """
        if self._expand != Expansion.ALL:
            return  # Only filter in ALL mode; INDEX renders headers itself

        child_ids = self._get_child_ids()

        # ALL mode: show only selected child, hide all if no selection
        for child_id in child_ids:
            if child_id in views:
                if self._selected_id is None:
                    views[child_id].hide = True
                else:
                    views[child_id].hide = (child_id != self._selected_id)
                # Selected child keeps its own expand mode

    def get_options(self) -> list[str]:
        """Get titles of all child options.

        Returns:
            List of child titles in child_order order
        """
        node = self._node
        graph = getattr(node, "_graph", None)
        if graph is None:
            return []

        # Get child IDs in order
        child_order = getattr(node, "child_order", None)
        if child_order is not None:
            if hasattr(child_order, "to_list"):
                child_ids = child_order.to_list()
            else:
                child_ids = list(child_order)
        else:
            child_ids = list(getattr(node, "children_ids", set()))

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
        node = self._node
        graph = getattr(node, "_graph", None)
        if graph is None:
            return ""

        child_ids = self._get_child_ids()
        lines = []
        for child_id in child_ids:
            child = graph.get_node(child_id)
            if child:
                # Get header line from child
                header = getattr(child, "render_header", None)
                if callable(header):
                    lines.append(header())
                else:
                    # Fallback to title or ID
                    title = getattr(child, "title", None) or child_id
                    lines.append(f"- {title}")
        return "\n".join(lines)

    def render_brief(self) -> str:
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
            node = self._node
            graph = getattr(node, "_graph", None)
            if graph:
                selected_child = graph.get_node(self._selected_id)
                if selected_child:
                    selected_title = getattr(selected_child, "title", None) or self._selected_id

        if selected_title is None:
            selected_title = options[0] if options else "?"

        return f"{selected_title} [{' | '.join(options)}]"

    def __repr__(self) -> str:
        """Return string representation."""
        hide_str = ", hide=True" if self._hide else ""
        selected_str = f", selected={self._selected_id!r}" if self._selected_id else ""
        return f"ChoiceView({self._node!r}, expand={self._expand.value}{hide_str}{selected_str})"


class SequenceView(ChoiceView):
    """View for ordered sequential progression through children.

    SequenceView extends ChoiceView to add progression semantics:
    - Agent works through steps in order
    - Current step is visible, others are hidden (like ChoiceView)
    - Tracks completion state per step
    - Supports forward/backward navigation

    State is persisted in node tags for session save/restore:
    - _seq_index: Current step index
    - _seq_completed: Set of completed step indices

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

    __slots__ = ("_current_index", "_completed_steps")

    _current_index: int
    _completed_steps: set[int]

    def __init__(
        self,
        node: ContextNode,
        selected_id: str | None = None,
        hide: bool = False,
        expand: Expansion | None = None,
    ) -> None:
        """Create a sequence view wrapping a node.

        Args:
            node: The ContextNode to wrap (typically a GroupNode)
            selected_id: Initial selection (default: first child)
            hide: Whether the view is hidden (default False)
            expand: Expansion state (default: ALL for full content)
        """
        # Default expand to ALL for sequences
        if expand is None:
            expand = Expansion.ALL

        super().__init__(node, selected_id=selected_id, hide=hide, expand=expand)

        # Initialize progression state
        object.__setattr__(self, "_current_index", 0)
        object.__setattr__(self, "_completed_steps", set())

        # Restore state from node tags if present (for session restore)
        node_tags = getattr(node, "tags", {})
        if "_seq_index" in node_tags:
            object.__setattr__(self, "_current_index", node_tags["_seq_index"])
        if "_seq_completed" in node_tags:
            object.__setattr__(
                self, "_completed_steps", set(node_tags["_seq_completed"])
            )

        # Set initial selection based on index
        child_ids = self._get_child_ids()
        if child_ids and self._selected_id is None and 0 <= self._current_index < len(child_ids):
            object.__setattr__(self, "_selected_id", child_ids[self._current_index])

    def _persist_state(self) -> None:
        """Persist progression state to node tags."""
        node_tags = getattr(self._node, "tags", None)
        if node_tags is not None:
            node_tags["_seq_index"] = self._current_index
            node_tags["_seq_completed"] = list(self._completed_steps)

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

        self._persist_state()
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
            self._persist_state()
        return self

    def mark_complete(self) -> SequenceView:
        """Mark current step as complete without advancing (fluent API).

        Returns:
            Self for method chaining
        """
        self._completed_steps.add(self._current_index)
        self._persist_state()
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
            self._persist_state()
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
            self._persist_state()
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
        node = self._node
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
            current_marker = " ← current" if i == self._current_index else ""
            lines.append(f"- {marker} Step {i + 1}: {title}{current_marker}")

        return "\n".join(lines)

    def __setattr__(self, name: str, value: Any) -> None:
        """Handle attribute assignment for sequence-specific slots."""
        if name in ("_current_index", "_completed_steps"):
            object.__setattr__(self, name, value)
        else:
            super().__setattr__(name, value)

    def __repr__(self) -> str:
        """Return string representation."""
        hide_str = ", hide=True" if self._hide else ""
        return f"SequenceView({self._node!r}, progress={self.progress}{hide_str})"


class LoopView(NodeView):
    """View for iterative refinement loops.

    LoopView wraps a single child node and tracks iteration state:
    - Counts iterations
    - Accumulates state across iterations
    - Supports early exit via done()
    - Optional max_iterations limit

    State is persisted in node tags for session save/restore:
    - _loop_iteration: Current iteration count (1-based)
    - _loop_state: Accumulated state dictionary
    - _loop_done: Whether loop was exited early

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

    __slots__ = ("_iteration", "_state", "_done", "_max_iterations")

    _iteration: int
    _state: dict[str, Any]
    _done: bool
    _max_iterations: int | None

    def __init__(
        self,
        node: ContextNode,
        max_iterations: int | None = None,
        hide: bool = False,
        expand: Expansion | None = None,
    ) -> None:
        """Create a loop view wrapping a node.

        Args:
            node: The ContextNode to wrap
            max_iterations: Maximum iterations allowed (None = unlimited)
            hide: Whether the view is hidden (default False)
            expand: Expansion state (default: ALL)
        """
        if expand is None:
            expand = Expansion.ALL

        super().__init__(node, hide=hide, expand=expand)

        object.__setattr__(self, "_iteration", 1)
        object.__setattr__(self, "_state", {})
        object.__setattr__(self, "_done", False)
        object.__setattr__(self, "_max_iterations", max_iterations)

        # Restore state from node tags if present (for session restore)
        node_tags = getattr(node, "tags", {})
        if "_loop_iteration" in node_tags:
            object.__setattr__(self, "_iteration", node_tags["_loop_iteration"])
        if "_loop_state" in node_tags:
            object.__setattr__(self, "_state", dict(node_tags["_loop_state"]))
        if "_loop_done" in node_tags:
            object.__setattr__(self, "_done", node_tags["_loop_done"])

    def _persist_state(self) -> None:
        """Persist loop state to node tags."""
        node_tags = getattr(self._node, "tags", None)
        if node_tags is not None:
            node_tags["_loop_iteration"] = self._iteration
            node_tags["_loop_state"] = dict(self._state)
            node_tags["_loop_done"] = self._done

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

        self._persist_state()
        return self

    def update_state(self, **state_updates: Any) -> LoopView:
        """Update state without incrementing iteration (fluent API).

        Args:
            **state_updates: Key-value pairs to merge into state

        Returns:
            Self for method chaining
        """
        self._state.update(state_updates)
        self._persist_state()
        return self

    def done(self) -> LoopView:
        """Mark loop as complete (fluent API).

        Returns:
            Self for method chaining
        """
        object.__setattr__(self, "_done", True)
        self._persist_state()
        return self

    def reset(self) -> LoopView:
        """Reset loop to initial state (fluent API).

        Returns:
            Self for method chaining
        """
        object.__setattr__(self, "_iteration", 1)
        object.__setattr__(self, "_state", {})
        object.__setattr__(self, "_done", False)
        self._persist_state()
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

        title = getattr(self._node, "title", None) or "Loop"
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
        """Handle attribute assignment for loop-specific slots."""
        if name in ("_iteration", "_state", "_done", "_max_iterations"):
            object.__setattr__(self, name, value)
        else:
            super().__setattr__(name, value)

    def __repr__(self) -> str:
        """Return string representation."""
        hide_str = ", hide=True" if self._hide else ""
        max_str = f"/{self._max_iterations}" if self._max_iterations else ""
        done_str = ", done" if self._done else ""
        return f"LoopView({self._node!r}, iteration={self._iteration}{max_str}{done_str}{hide_str})"


class StateView(ChoiceView):
    """View for state machine navigation through named states.

    StateView extends ChoiceView to add state machine semantics:
    - Named states mapped to child nodes
    - Transition rules defining valid state changes
    - Current state visible, others hidden
    - State history tracking

    State is persisted in node tags for session save/restore:
    - _state_current: Current state name
    - _state_history: List of previous states

    Rendering:
    - Header shows current state and available transitions
    - "## Task State: working → [done, idle]"

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

    __slots__ = ("_states", "_transitions", "_current_state", "_state_history")

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
        hide: bool = False,
        expand: Expansion | None = None,
    ) -> None:
        """Create a state machine view wrapping a node.

        Args:
            node: The ContextNode to wrap (typically a GroupNode)
            states: Mapping of state names to child node IDs
            transitions: Mapping of state names to allowed next states
            initial: Initial state name (default: first state)
            hide: Whether the view is hidden (default False)
            expand: Expansion state (default: ALL)
        """
        if expand is None:
            expand = Expansion.ALL

        # Initialize states and transitions
        states = states or {}
        transitions = transitions or {}

        # Default initial state to first key
        if initial is None and states:
            initial = next(iter(states.keys()))

        # Get initial node ID for ChoiceView selection
        initial_node_id = states.get(initial) if initial else None

        super().__init__(node, selected_id=initial_node_id, hide=hide, expand=expand)

        object.__setattr__(self, "_states", states)
        object.__setattr__(self, "_transitions", transitions)
        object.__setattr__(self, "_current_state", initial or "")
        object.__setattr__(self, "_state_history", [])

        # Restore state from node tags if present (for session restore)
        node_tags = getattr(node, "tags", {})
        if "_state_current" in node_tags:
            current = node_tags["_state_current"]
            object.__setattr__(self, "_current_state", current)
            # Sync selection with restored state
            if current in self._states:
                object.__setattr__(self, "_selected_id", self._states[current])
        if "_state_history" in node_tags:
            object.__setattr__(self, "_state_history", list(node_tags["_state_history"]))

    def _persist_state(self) -> None:
        """Persist state machine state to node tags."""
        node_tags = getattr(self._node, "tags", None)
        if node_tags is not None:
            node_tags["_state_current"] = self._current_state
            node_tags["_state_history"] = list(self._state_history)

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

        self._persist_state()
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

        self._persist_state()
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
        self._persist_state()
        return self

    def render_header(self) -> str:
        """Render state header with current state and transitions.

        Returns:
            Header like "## Task State: working → [done, idle]"
        """
        title = getattr(self._node, "title", None) or "State"
        transitions = self.valid_transitions
        trans_str = f" → [{', '.join(transitions)}]" if transitions else " (terminal)"
        return f"## {title}: {self._current_state}{trans_str}"

    def __setattr__(self, name: str, value: Any) -> None:
        """Handle attribute assignment for state-specific slots."""
        if name in ("_states", "_transitions", "_current_state", "_state_history"):
            object.__setattr__(self, name, value)
        else:
            super().__setattr__(name, value)

    def __repr__(self) -> str:
        """Return string representation."""
        hide_str = ", hide=True" if self._hide else ""
        history_len = len(self._state_history)
        history_str = f", history={history_len}" if history_len else ""
        return f"StateView({self._node!r}, state={self._current_state!r}{history_str}{hide_str})"

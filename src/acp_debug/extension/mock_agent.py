"""Mock agent base class for testing IDE integrations."""

from __future__ import annotations

import uuid
from collections.abc import Awaitable, Callable

from acp_debug.extension.base import ACPExtension
from acp_debug.types.common import (
    AgentCapabilities,
    AgentInfo,
    ModeInfo,
    ModelInfo,
    PromptCapabilities,
)
from acp_debug.types.requests import (
    InitializeRequest,
    ListSessionsRequest,
    LoadSessionRequest,
    NewSessionRequest,
    PromptRequest,
    SetModelRequest,
    SetModeRequest,
)
from acp_debug.types.responses import (
    InitializeResponse,
    ListSessionsResponse,
    LoadSessionResponse,
    ModelsInfo,
    ModesInfo,
    NewSessionResponse,
    PromptResponse,
    SetModelResponse,
    SetModeResponse,
    StopReason,
)


class MockAgentBase(ACPExtension):
    """Base class for mock agents - override methods to customize responses.

    Default behavior provides sensible stub responses for all methods.

    Usage:
        class MyMockAgent(MockAgentBase):
            async def session_prompt(self, req, call):
                # Custom prompt handling
                await self.emit_text(req.session_id, "Hello from mock agent!")
                return PromptResponse(stop_reason=StopReason.END_TURN)
    """

    def __init__(self) -> None:
        super().__init__()
        self.name = "mock-agent"
        self.version = "1.0.0"

    def _generate_session_id(self) -> str:
        """Generate a unique session ID."""
        return f"mock-{uuid.uuid4().hex[:8]}"

    async def initialize(
        self,
        req: InitializeRequest,
        call: Callable[[InitializeRequest], Awaitable[InitializeResponse]],
    ) -> InitializeResponse:
        """Return mock agent capabilities."""
        return InitializeResponse(
            protocol_version=req.protocol_version,
            agent_capabilities=AgentCapabilities(
                load_session=True,
                prompt_capabilities=PromptCapabilities(
                    image=False,
                    audio=False,
                    embedded_context=True,
                ),
            ),
            agent_info=AgentInfo(
                name=self.name,
                title="Mock Agent",
                version=self.version,
            ),
        )

    async def session_new(
        self,
        req: NewSessionRequest,
        call: Callable[[NewSessionRequest], Awaitable[NewSessionResponse]],
    ) -> NewSessionResponse:
        """Create a new mock session."""
        session_id = self._generate_session_id()
        self.on_session_bind(session_id)
        self.sessions[session_id].cwd = req.cwd

        return NewSessionResponse(
            session_id=session_id,
            models=ModelsInfo(
                available_models=[
                    ModelInfo(
                        model_id="mock-model",
                        name="Mock Model",
                        description="A mock model for testing",
                    )
                ],
                current_model_id="mock-model",
            ),
            modes=ModesInfo(
                available_modes=[
                    ModeInfo(id="normal", name="Normal", description="Standard mode"),
                ],
                current_mode_id="normal",
            ),
        )

    async def session_load(
        self,
        req: LoadSessionRequest,
        call: Callable[[LoadSessionRequest], Awaitable[LoadSessionResponse]],
    ) -> LoadSessionResponse:
        """Load an existing mock session."""
        self.on_session_bind(req.session_id)
        self.sessions[req.session_id].cwd = req.cwd

        return LoadSessionResponse(
            session_id=req.session_id,
            models=ModelsInfo(
                available_models=[
                    ModelInfo(
                        model_id="mock-model",
                        name="Mock Model",
                        description="A mock model for testing",
                    )
                ],
                current_model_id="mock-model",
            ),
            modes=ModesInfo(
                available_modes=[
                    ModeInfo(id="normal", name="Normal", description="Standard mode"),
                ],
                current_mode_id="normal",
            ),
        )

    async def session_list(
        self,
        req: ListSessionsRequest,
        call: Callable[[ListSessionsRequest], Awaitable[ListSessionsResponse]],
    ) -> ListSessionsResponse:
        """List mock sessions (empty by default)."""
        return ListSessionsResponse(sessions=[])

    async def session_prompt(
        self,
        req: PromptRequest,
        call: Callable[[PromptRequest], Awaitable[PromptResponse]],
    ) -> PromptResponse:
        """Handle prompt - override this for custom behavior."""
        # Default: immediate end_turn with no content
        return PromptResponse(stop_reason=StopReason.END_TURN)

    async def session_set_mode(
        self,
        req: SetModeRequest,
        call: Callable[[SetModeRequest], Awaitable[SetModeResponse]],
    ) -> SetModeResponse:
        """Set session mode."""
        if req.session_id in self.sessions:
            self.sessions[req.session_id].mode = req.mode_id
        return SetModeResponse(mode_id=req.mode_id)

    async def session_set_model(
        self,
        req: SetModelRequest,
        call: Callable[[SetModelRequest], Awaitable[SetModelResponse]],
    ) -> SetModelResponse:
        """Set session model."""
        if req.session_id in self.sessions:
            self.sessions[req.session_id].model = req.model_id
        return SetModelResponse(model_id=req.model_id)

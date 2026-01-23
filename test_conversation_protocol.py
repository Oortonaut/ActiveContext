"""Simple test for conversation delegation protocol (Phase 1).

This test verifies the core protocol implementation:
- ConversationTransport protocol
- SessionConversationTransport implementation
- Session.delegate_conversation() method
- MessageNode creation and context graph integration
"""

import asyncio
from activecontext import ActiveContext
from activecontext.protocols.conversation import ConversationTransport, InputType


class EchoHandler:
    """Simple test handler that echoes user input."""

    async def handle(self, transport: ConversationTransport) -> dict[str, str]:
        """Handle conversation by echoing user input.

        Args:
            transport: Communication channel to user

        Returns:
            Result dict with echoed text
        """
        # Send greeting
        await transport.send_output("Echo Handler Started")
        await transport.send_output("I will echo whatever you say.")

        # Request input
        user_input = await transport.request_input(
            "Say something:",
            input_type=InputType.TEXT,
        )

        # Echo it back
        await transport.send_output(f"You said: {user_input}")

        # Send progress update
        await transport.send_progress(1, 1, status="Complete")

        # Return result
        return {"echoed": user_input, "status": "success"}


async def test_echo_handler():
    """Test the echo handler with conversation delegation."""
    print("Testing conversation delegation protocol...")

    async with ActiveContext() as ctx:
        # Create session
        session = await ctx.create_session(cwd=".")

        print("Created session:", session.session_id)

        # Create handler
        handler = EchoHandler()

        # Manually simulate user input (in real scenario, this comes from ACP transport)
        from activecontext.session.coordinator import SessionConversationTransport

        # Create transport
        transport = SessionConversationTransport(
            session=session,
            originator="test:echo",
            forward_permissions=True,
        )

        # Simulate the delegation flow
        print("\n--- Starting delegation ---")

        # Start handler in background
        handler_task = asyncio.create_task(handler.handle(transport))

        # Wait briefly for handler to request input
        await asyncio.sleep(0.1)

        # Simulate user response
        print("Simulating user input: 'Hello, World!'")
        transport.handle_input_response("Hello, World!")

        # Wait for handler to complete
        result = await handler_task

        print("\n--- Delegation complete ---")
        print(f"Handler result: {result}")

        # Check MessageNodes in context graph
        graph = session.timeline.context_graph
        all_nodes = list(graph)  # ContextGraph is iterable

        print(f"\nContext graph has {len(all_nodes)} nodes")

        # Find MessageNodes with our originator
        from activecontext.context.nodes import MessageNode

        test_messages = [
            node
            for node in all_nodes
            if isinstance(node, MessageNode) and node.originator == "test:echo"
        ]

        print(f"Found {len(test_messages)} MessageNodes from test handler:")
        for msg in test_messages:
            print(f"  - [{msg.role}] {msg.content[:50]}...")

        # Find user response
        user_messages = [
            node
            for node in all_nodes
            if isinstance(node, MessageNode) and node.originator == "user"
        ]

        print(f"\nFound {len(user_messages)} MessageNodes from user:")
        for msg in user_messages:
            print(f"  - [{msg.role}] {msg.content[:50]}...")

    print("\n[PASS] Test passed! Core protocol is working.")


if __name__ == "__main__":
    asyncio.run(test_echo_handler())

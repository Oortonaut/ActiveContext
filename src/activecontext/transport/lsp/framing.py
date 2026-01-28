"""LSP message framing with Content-Length headers.

This module implements the LSP base protocol framing:
- Header parsing (Content-Length required, Content-Type optional)
- Message reading with proper buffering for partial messages
- Message writing with Content-Length framing

LSP Header Format:
    Content-Length: <length>\r\n
    [Content-Type: <type>]\r\n
    \r\n
    <json-rpc-message>

The Content-Length header is required and specifies the byte count
of the JSON-RPC message body. Headers are separated from the body
by a blank line (double CRLF).
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

# Header constants
CONTENT_LENGTH = b"Content-Length"
CONTENT_TYPE = b"Content-Type"
HEADER_ENCODING = "ascii"
CONTENT_ENCODING = "utf-8"
CRLF = b"\r\n"
HEADER_SEPARATOR = b"\r\n\r\n"


class LSPFramingError(Exception):
    """Error in LSP message framing.

    Raised when:
    - Content-Length header is missing
    - Content-Length value is not a valid integer
    - Content-Length value is negative
    - Header format is malformed
    - JSON content cannot be parsed
    """

    pass


def parse_header(header_bytes: bytes) -> dict[str, str]:
    """Parse LSP headers from raw bytes.

    Args:
        header_bytes: Raw header bytes (without trailing CRLF CRLF separator).
            Should contain lines like "Content-Length: 123\r\nContent-Type: ..."

    Returns:
        Dictionary mapping header names to values.
        Keys are normalized to their canonical form (e.g., "Content-Length").

    Raises:
        LSPFramingError: If headers are malformed or Content-Length is missing/invalid.

    Example:
        >>> header = b"Content-Length: 42\\r\\nContent-Type: application/json"
        >>> parse_header(header)
        {'Content-Length': '42', 'Content-Type': 'application/json'}
    """
    headers: dict[str, str] = {}

    if not header_bytes:
        raise LSPFramingError("Empty header block")

    try:
        header_text = header_bytes.decode(HEADER_ENCODING)
    except UnicodeDecodeError as e:
        raise LSPFramingError(f"Header contains non-ASCII characters: {e}") from e

    # Split into individual header lines
    lines = header_text.split("\r\n")

    for line in lines:
        if not line:
            continue

        # Each header line is "Name: Value"
        colon_pos = line.find(":")
        if colon_pos == -1:
            raise LSPFramingError(f"Malformed header line (no colon): {line!r}")

        name = line[:colon_pos].strip()
        value = line[colon_pos + 1 :].strip()

        if not name:
            raise LSPFramingError(f"Empty header name in line: {line!r}")

        headers[name] = value

    # Validate required Content-Length header
    if "Content-Length" not in headers:
        raise LSPFramingError("Missing required Content-Length header")

    try:
        length = int(headers["Content-Length"])
    except ValueError as e:
        raise LSPFramingError(f"Invalid Content-Length value: {headers['Content-Length']!r}") from e

    if length < 0:
        raise LSPFramingError(f"Negative Content-Length: {length}")

    return headers


async def read_message(
    reader: asyncio.StreamReader,
    *,
    max_message_size: int = 10 * 1024 * 1024,  # 10 MB default
) -> dict[str, Any] | None:
    """Read a single LSP message from the stream.

    Handles partial message buffering automatically. Blocks until a complete
    message is available or EOF is reached.

    Args:
        reader: Async stream reader to read from.
        max_message_size: Maximum allowed message size in bytes.
            Prevents memory exhaustion from malformed Content-Length.

    Returns:
        Parsed JSON-RPC message as a dictionary, or None on EOF.

    Raises:
        LSPFramingError: If message framing is invalid or JSON parsing fails.

    Example:
        >>> async with open_lsp_connection() as (reader, writer):
        ...     while (msg := await read_message(reader)) is not None:
        ...         handle_message(msg)
    """
    # Read headers line by line until we hit the empty line separator
    header_bytes = b""

    while True:
        try:
            line = await reader.readuntil(CRLF)
        except asyncio.IncompleteReadError as e:
            # EOF before complete header line
            if header_bytes == b"" and reader.at_eof():
                return None  # Clean EOF at message boundary
            raise LSPFramingError("Unexpected EOF while reading headers") from e
        except asyncio.LimitOverrunError as e:
            raise LSPFramingError(f"Header line too long: {e}") from e

        if line == CRLF:
            # Empty line signals end of headers
            break

        header_bytes += line

    # Strip trailing CRLF from accumulated headers
    if header_bytes.endswith(CRLF):
        header_bytes = header_bytes[:-2]

    # Parse headers
    headers = parse_header(header_bytes)
    content_length = int(headers["Content-Length"])

    # Validate message size
    if content_length > max_message_size:
        raise LSPFramingError(f"Message size {content_length} exceeds maximum {max_message_size}")

    # Read the message body
    try:
        body_bytes = await reader.readexactly(content_length)
    except asyncio.IncompleteReadError as e:
        raise LSPFramingError(
            f"Incomplete message body: expected {content_length} bytes, got {len(e.partial)}"
        ) from e

    # Parse JSON content
    try:
        content = body_bytes.decode(CONTENT_ENCODING)
    except UnicodeDecodeError as e:
        raise LSPFramingError(f"Invalid UTF-8 in message body: {e}") from e

    try:
        message = json.loads(content)
    except json.JSONDecodeError as e:
        raise LSPFramingError(f"Invalid JSON in message body: {e}") from e

    if not isinstance(message, dict):
        raise LSPFramingError(f"JSON-RPC message must be an object, got {type(message).__name__}")

    return message


async def write_message(
    writer: asyncio.StreamWriter,
    msg: dict[str, Any],
    *,
    drain: bool = True,
) -> None:
    """Write a JSON-RPC message with Content-Length framing.

    Args:
        writer: Async stream writer to write to.
        msg: JSON-RPC message to send (must be a JSON-serializable dict).
        drain: If True, wait for the message to be flushed to the OS buffer.

    Raises:
        LSPFramingError: If the message cannot be serialized to JSON.

    Example:
        >>> await write_message(writer, {
        ...     "jsonrpc": "2.0",
        ...     "id": 1,
        ...     "result": {"capabilities": {}}
        ... })
    """
    try:
        # Serialize to compact JSON
        body = json.dumps(msg, separators=(",", ":"))
        body_bytes = body.encode(CONTENT_ENCODING)
    except (TypeError, ValueError) as e:
        raise LSPFramingError(f"Message cannot be serialized to JSON: {e}") from e

    # Build headers
    content_length = len(body_bytes)
    header = f"Content-Length: {content_length}\r\n\r\n"
    header_bytes = header.encode(HEADER_ENCODING)

    # Write complete message atomically (header + body)
    writer.write(header_bytes + body_bytes)

    if drain:
        await writer.drain()

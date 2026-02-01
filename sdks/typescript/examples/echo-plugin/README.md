# Echo Plugin Example

A minimal CAP plugin server that echoes input text. Demonstrates the full plugin API.

## What It Does

The echo plugin provides a single node type (`echo`) that:

- Accepts a text string and optional repeat count.
- Renders the text at three granularity levels (header, content, detail).
- Supports method calls to update the text and count.
- Reports token estimates and digest metadata.

## Running

```bash
# From the sdks/typescript directory
npm install
npx ts-node examples/echo-plugin/index.ts
```

## Configuring in ActiveContext

Add to your project's `.ac/config.yaml`:

```yaml
plugins:
  - name: echo
    command: ["npx", "ts-node", "path/to/examples/echo-plugin/index.ts"]
```

Then in the DSL:

```python
e = echo("Hello, world!", echo_count=3)
# Renders "Hello, world!" three times

e.set_text("Updated text")
e.set_count(5)
```

## Node Type Schema

| Component | Details |
|-----------|---------|
| **Type** | `echo` |
| **Constructor** | `echo(text, echo_count=1)` |
| **Properties** | `text` (str, read-only), `echo_count` (int, read-only) |
| **Methods** | `set_text(text)`, `set_count(count)`, `get_stats()` |

## Code Structure

- `index.ts` -- Complete plugin in a single file:
  - `EchoNode` class extending `BaseNode`.
  - Schema definition using helper functions.
  - `CAPServer` setup and registration.

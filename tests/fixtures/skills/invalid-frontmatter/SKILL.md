---
name: invalid-frontmatter
description: Test skill with malformed YAML
allowed-tools: [unclosed bracket
metadata:
  version: "1.0.0
  unterminated: string
---

# Invalid Frontmatter

This skill has malformed YAML in the frontmatter.
It should fail to load.

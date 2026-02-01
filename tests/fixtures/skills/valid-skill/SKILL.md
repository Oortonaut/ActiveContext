---
name: valid-skill
description: A well-formed test skill with all required fields
license: MIT
allowed-tools:
  - Bash
  - Read
  - Write
metadata:
  version: 1.0.0
  author: Test Suite
  model: claude-opus-4-5-20251101
---

# Valid Skill

This is a valid skill fixture for testing skill loading infrastructure.

## Purpose

Used to verify that the skill loader correctly handles well-formed SKILL.md files.

## Features

- Proper YAML frontmatter
- All required fields (name, description)
- Optional fields (license, allowed-tools, metadata)
- Clean markdown content

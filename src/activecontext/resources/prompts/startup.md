# Session Startup

Default startup script for new ActiveContext sessions.
Configures reference documentation and mode scripts.

## Reference Documentation

```python/acrepl
markdown("@prompts/context_guide.md", expansion=Expansion.ALL)
# markdown("@prompts/dsl_reference.md", expansion=Expansion.ALL)
markdown("@prompts/node_states.md", expansion=Expansion.ALL)
markdown("@prompts/context_graph.md", expansion=Expansion.ALL)
markdown("@prompts/work_coordination.md", expansion=Expansion.ALL)
markdown("@prompts/mcp.md", expansion=Expansion.ALL)
```

## Mode Scripts

```python/acrepl
# _mode_normal = markdown("@prompts/modes/normal.md")
# _mode_plan = markdown("@prompts/modes/plan.md")
# _mode_brave = markdown("@prompts/modes/brave.md")
# _mode_scripts = choice(_mode_normal, _mode_plan, _mode_brave, selected=_mode_normal.node_id)
# __session__.set_mode_choice_view(_mode_scripts)
```

#!/usr/bin/env python3
"""Static call graph generator for Python projects.

Parses Python source files using the ast module and extracts:
- Module/class/function definitions (the "defines" graph)
- Function-to-function call relationships (the "uses" graph)

Outputs:
- docs/callgraph/defines.md   — Hierarchical module structure
- docs/callgraph/calls.md     — Call relationships per module
- docs/callgraph/summary.md   — High-level cross-module dependencies

Usage:
    python scripts/callgraph.py [--src SRC_DIR] [--out OUT_DIR]
"""

from __future__ import annotations

import argparse
import ast
import os
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class FunctionDef:
    """A function or method definition."""

    name: str
    qualname: str  # e.g. "ContextGraph.add_node"
    module: str  # e.g. "activecontext.context.graph"
    lineno: int
    is_method: bool = False
    is_async: bool = False
    calls: list[str] = field(default_factory=list)  # qualnames of called functions
    decorators: list[str] = field(default_factory=list)


@dataclass
class ClassDef:
    """A class definition."""

    name: str
    qualname: str
    module: str
    lineno: int
    bases: list[str] = field(default_factory=list)
    methods: list[str] = field(default_factory=list)  # method qualnames


@dataclass
class ModuleInfo:
    """Collected information about a single module."""

    name: str  # dotted module name
    path: str  # relative file path
    imports: list[str] = field(default_factory=list)  # imported module names
    classes: list[ClassDef] = field(default_factory=list)
    functions: list[FunctionDef] = field(default_factory=list)


class CallGraphVisitor(ast.NodeVisitor):
    """AST visitor that extracts definitions and call relationships."""

    def __init__(self, module_name: str) -> None:
        self.module = module_name
        self._scope: list[str] = []  # current class/function nesting
        self.classes: list[ClassDef] = []
        self.functions: list[FunctionDef] = []
        self.imports: list[str] = []
        self._current_func: FunctionDef | None = None

    def _qualname(self, name: str) -> str:
        parts = [self.module] + self._scope + [name]
        return ".".join(parts)

    def _resolve_call_name(self, node: ast.expr) -> str | None:
        """Extract a readable name from a call target."""
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute):
            value = self._resolve_call_name(node.value)
            if value:
                return f"{value}.{node.attr}"
            return node.attr
        if isinstance(node, ast.Call):
            return self._resolve_call_name(node.func)
        return None

    def _get_decorator_names(self, decorator_list: list[ast.expr]) -> list[str]:
        names = []
        for d in decorator_list:
            name = self._resolve_call_name(d)
            if name:
                names.append(name)
        return names

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            self.imports.append(alias.name)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if node.module:
            self.imports.append(node.module)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        bases = []
        for base in node.bases:
            name = self._resolve_call_name(base)
            if name:
                bases.append(name)

        cls = ClassDef(
            name=node.name,
            qualname=self._qualname(node.name),
            module=self.module,
            lineno=node.lineno,
            bases=bases,
        )
        self.classes.append(cls)

        self._scope.append(node.name)
        self.generic_visit(node)
        self._scope.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._visit_func(node, is_async=False)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._visit_func(node, is_async=True)

    def _visit_func(self, node: ast.FunctionDef | ast.AsyncFunctionDef, is_async: bool) -> None:
        func = FunctionDef(
            name=node.name,
            qualname=self._qualname(node.name),
            module=self.module,
            lineno=node.lineno,
            is_method=len(self._scope) > 0 and not isinstance(node, ast.Module),
            is_async=is_async,
            decorators=self._get_decorator_names(node.decorator_list),
        )
        self.functions.append(func)

        # If inside a class, register as method
        if self._scope:
            for cls in reversed(self.classes):
                if cls.name == self._scope[-1]:
                    cls.methods.append(func.qualname)
                    break

        # Track calls within this function
        outer = self._current_func
        self._current_func = func
        self._scope.append(node.name)
        self.generic_visit(node)
        self._scope.pop()
        self._current_func = outer

    def visit_Call(self, node: ast.Call) -> None:
        if self._current_func is not None:
            name = self._resolve_call_name(node.func)
            if name:
                self._current_func.calls.append(name)
        self.generic_visit(node)


def get_module_name(filepath: Path, src_root: Path) -> str:
    """Convert a file path to a dotted module name."""
    rel = filepath.relative_to(src_root)
    parts = list(rel.parts)
    if parts[-1] == "__init__.py":
        parts = parts[:-1]
    else:
        parts[-1] = parts[-1].removesuffix(".py")
    return ".".join(parts)


def analyze_file(filepath: Path, src_root: Path) -> ModuleInfo | None:
    """Parse a single Python file and extract call graph info."""
    try:
        source = filepath.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(filepath))
    except (SyntaxError, UnicodeDecodeError) as e:
        print(f"  Warning: skipping {filepath}: {e}", file=sys.stderr)
        return None

    module_name = get_module_name(filepath, src_root)
    visitor = CallGraphVisitor(module_name)
    visitor.visit(tree)

    return ModuleInfo(
        name=module_name,
        path=str(filepath.relative_to(src_root.parent)),
        imports=visitor.imports,
        classes=visitor.classes,
        functions=visitor.functions,
    )


def collect_modules(src_dir: Path) -> list[ModuleInfo]:
    """Walk the source tree and analyze all Python files."""
    modules = []
    for root, _dirs, files in os.walk(src_dir):
        for fname in sorted(files):
            if not fname.endswith(".py"):
                continue
            filepath = Path(root) / fname
            info = analyze_file(filepath, src_dir)
            if info:
                modules.append(info)
    return modules


def write_defines(modules: list[ModuleInfo], out_dir: Path) -> None:
    """Write defines.md — hierarchical module/class/function structure."""
    lines: list[str] = [
        "# Module Structure",
        "",
        "Hierarchical view of all modules, classes, and functions.",
        "Generated by `scripts/callgraph.py`.",
        "",
    ]

    for mod in modules:
        lines.append(f"## `{mod.name}`")
        lines.append(f"File: `{mod.path}`")
        lines.append("")

        # Top-level functions (not methods)
        method_qualnames = set()
        for cls in mod.classes:
            method_qualnames.update(cls.methods)

        top_funcs = [f for f in mod.functions if f.qualname not in method_qualnames]

        if top_funcs:
            lines.append("### Functions")
            lines.append("")
            for func in top_funcs:
                prefix = "async " if func.is_async else ""
                deco = ""
                if func.decorators:
                    deco = f" `@{'`, `@'.join(func.decorators)}`"
                lines.append(f"- `{prefix}{func.name}` (line {func.lineno}){deco}")
            lines.append("")

        for cls in mod.classes:
            bases_str = ""
            if cls.bases:
                bases_str = f"({', '.join(cls.bases)})"
            lines.append(f"### class `{cls.name}{bases_str}`")
            lines.append("")
            cls_methods = [f for f in mod.functions if f.qualname in set(cls.methods)]
            for func in cls_methods:
                prefix = "async " if func.is_async else ""
                lines.append(f"- `{prefix}{func.name}` (line {func.lineno})")
            lines.append("")

    (out_dir / "defines.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"  Wrote {out_dir / 'defines.md'} ({len(modules)} modules)")


def write_calls(modules: list[ModuleInfo], out_dir: Path) -> None:
    """Write calls.md — call relationships grouped by module."""
    lines: list[str] = [
        "# Call Graph",
        "",
        "Function-to-function call relationships extracted via static analysis.",
        "Generated by `scripts/callgraph.py`.",
        "",
        "> **Note:** Static analysis captures direct name references in source code.",
        "> It may miss dynamic dispatch, callbacks, and calls through variables.",
        "> It may include false positives from shadowed names or conditional branches.",
        "",
    ]

    for mod in modules:
        funcs_with_calls = [f for f in mod.functions if f.calls]
        if not funcs_with_calls:
            continue

        lines.append(f"## `{mod.name}`")
        lines.append("")

        for func in funcs_with_calls:
            prefix = "async " if func.is_async else ""
            lines.append(f"### `{prefix}{func.qualname}` (line {func.lineno})")
            lines.append("")

            # Deduplicate and sort calls
            seen: dict[str, int] = {}
            for call in func.calls:
                seen[call] = seen.get(call, 0) + 1

            for call, count in sorted(seen.items()):
                suffix = f" ×{count}" if count > 1 else ""
                lines.append(f"- `{call}`{suffix}")
            lines.append("")

    (out_dir / "calls.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"  Wrote {out_dir / 'calls.md'}")


def write_summary(modules: list[ModuleInfo], out_dir: Path) -> None:
    """Write summary.md — cross-module dependency overview."""
    lines: list[str] = [
        "# Cross-Module Dependencies",
        "",
        "Which modules import which. Internal imports only.",
        "Generated by `scripts/callgraph.py`.",
        "",
    ]

    # Build set of known module names for filtering
    known_modules = {mod.name for mod in modules}
    # Also add parent packages
    for mod in modules:
        parts = mod.name.split(".")
        for i in range(1, len(parts)):
            known_modules.add(".".join(parts[:i]))

    # Collect statistics
    total_modules = len(modules)
    total_classes = sum(len(m.classes) for m in modules)
    total_functions = sum(len(m.functions) for m in modules)
    total_calls = sum(sum(len(f.calls) for f in m.functions) for m in modules)

    lines.append("## Statistics")
    lines.append("")
    lines.append(f"| Metric | Count |")
    lines.append(f"|--------|-------|")
    lines.append(f"| Modules | {total_modules} |")
    lines.append(f"| Classes | {total_classes} |")
    lines.append(f"| Functions/Methods | {total_functions} |")
    lines.append(f"| Call references | {total_calls} |")
    lines.append("")

    # Group modules by top-level package
    packages: dict[str, list[ModuleInfo]] = defaultdict(list)
    for mod in modules:
        pkg = mod.name.split(".")[0]
        packages[pkg].append(mod)

    lines.append("## Package Overview")
    lines.append("")
    for pkg_name, pkg_modules in sorted(packages.items()):
        pkg_classes = sum(len(m.classes) for m in pkg_modules)
        pkg_funcs = sum(len(m.functions) for m in pkg_modules)
        lines.append(
            f"- **{pkg_name}**: {len(pkg_modules)} modules, "
            f"{pkg_classes} classes, {pkg_funcs} functions"
        )
    lines.append("")

    # Cross-module imports
    lines.append("## Import Graph")
    lines.append("")
    lines.append("Internal imports between modules (external dependencies excluded).")
    lines.append("")

    for mod in modules:
        internal_imports = sorted(set(
            imp for imp in mod.imports
            if any(imp.startswith(known) for known in
                   {m.name.split(".")[0] for m in modules})
        ))
        if internal_imports:
            lines.append(f"**`{mod.name}`** imports:")
            for imp in internal_imports:
                lines.append(f"- `{imp}`")
            lines.append("")

    # Cross-package edges (higher level)
    lines.append("## Cross-Package Dependencies")
    lines.append("")
    cross_pkg: dict[str, set[str]] = defaultdict(set)
    for mod in modules:
        src_pkg = mod.name.split(".")[0]
        for imp in mod.imports:
            dst_pkg = imp.split(".")[0]
            if dst_pkg != src_pkg and dst_pkg in packages:
                cross_pkg[src_pkg].add(dst_pkg)

    if cross_pkg:
        for src, dsts in sorted(cross_pkg.items()):
            lines.append(f"- **{src}** → {', '.join(f'**{d}**' for d in sorted(dsts))}")
    else:
        lines.append("No cross-package dependencies found.")
    lines.append("")

    (out_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"  Wrote {out_dir / 'summary.md'}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate static call graph for Python project")
    parser.add_argument("--src", default="src", help="Source directory (default: src)")
    parser.add_argument("--out", default="docs/callgraph", help="Output directory (default: docs/callgraph)")
    args = parser.parse_args()

    src_dir = Path(args.src).resolve()
    out_dir = Path(args.out)

    if not src_dir.is_dir():
        print(f"Error: source directory not found: {src_dir}", file=sys.stderr)
        sys.exit(1)

    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Analyzing {src_dir}...")
    modules = collect_modules(src_dir)
    print(f"  Found {len(modules)} modules")
    print()

    print("Generating output...")
    write_defines(modules, out_dir)
    write_calls(modules, out_dir)
    write_summary(modules, out_dir)
    print()
    print("Done.")


if __name__ == "__main__":
    main()

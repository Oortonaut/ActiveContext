#!/usr/bin/env python3
"""Render PlantUML diagrams to PNG and SVG.

Usage:
    uv run docs/diagrams/render.py              # Render all diagrams
    uv run docs/diagrams/render.py --png        # PNG only
    uv run docs/diagrams/render.py --svg        # SVG only
    uv run docs/diagrams/render.py acp/         # Render specific directory
    uv run docs/diagrams/render.py --clean      # Remove generated images
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import plantuml


def find_puml_files(base_dir: Path, subdirs: list[str] | None = None) -> list[Path]:
    """Find all .puml files in the given directories."""
    if subdirs:
        files = []
        for subdir in subdirs:
            subpath = base_dir / subdir
            if subpath.is_file() and subpath.suffix == ".puml":
                files.append(subpath)
            elif subpath.is_dir():
                files.extend(sorted(subpath.glob("*.puml")))
            else:
                print(f"Warning: {subpath} not found", file=sys.stderr)
        return files
    else:
        # Find in all subdirectories
        return sorted(base_dir.glob("**/*.puml"))


def render_diagram(
    puml_file: Path,
    server: plantuml.PlantUML,
    formats: list[str],
) -> bool:
    """Render a single diagram to the specified formats."""
    success = True
    
    for fmt in formats:
        output_file = puml_file.with_suffix(f".{fmt}")
        try:
            # Read the PlantUML source
            source = puml_file.read_text(encoding="utf-8")
            
            # Get the rendered image from the server
            if fmt == "png":
                url = server.get_url(source)
            else:  # svg
                # Modify URL for SVG output
                url = server.get_url(source).replace("/png/", "/svg/")
            
            # Fetch and save
            import urllib.request
            with urllib.request.urlopen(url) as response:
                output_file.write_bytes(response.read())
            
            print(f"  {output_file.name}")
            
        except Exception as e:
            print(f"  ERROR: {output_file.name} - {e}", file=sys.stderr)
            success = False
    
    return success


def clean_generated(base_dir: Path) -> None:
    """Remove all generated PNG and SVG files."""
    for ext in ("png", "svg"):
        for f in base_dir.glob(f"**/*.{ext}"):
            print(f"Removing {f.relative_to(base_dir)}")
            f.unlink()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Render PlantUML diagrams to PNG/SVG",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "paths",
        nargs="*",
        help="Specific subdirectories or files to render (default: all)",
    )
    parser.add_argument(
        "--png",
        action="store_true",
        help="Render PNG only",
    )
    parser.add_argument(
        "--svg",
        action="store_true",
        help="Render SVG only",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Remove generated images instead of rendering",
    )
    parser.add_argument(
        "--server",
        default="http://www.plantuml.com/plantuml/png/",
        help="PlantUML server URL (default: public server)",
    )
    
    args = parser.parse_args()
    
    # Determine base directory (where this script lives)
    base_dir = Path(__file__).parent
    
    if args.clean:
        clean_generated(base_dir)
        return 0
    
    # Determine formats
    if args.png and not args.svg:
        formats = ["png"]
    elif args.svg and not args.png:
        formats = ["svg"]
    else:
        formats = ["png", "svg"]
    
    # Find files to render
    puml_files = find_puml_files(base_dir, args.paths if args.paths else None)
    
    if not puml_files:
        print("No .puml files found", file=sys.stderr)
        return 1
    
    # Create PlantUML server connection
    server = plantuml.PlantUML(url=args.server)
    
    print(f"Rendering {len(puml_files)} diagrams to {', '.join(formats)}...")
    print(f"Using server: {args.server}\n")
    
    errors = 0
    current_dir = None
    
    for puml_file in puml_files:
        # Print directory header when it changes
        rel_dir = puml_file.parent.relative_to(base_dir)
        if rel_dir != current_dir:
            current_dir = rel_dir
            print(f"{rel_dir}/")
        
        if not render_diagram(puml_file, server, formats):
            errors += 1
    
    print(f"\nDone. {len(puml_files)} diagrams, {errors} errors.")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())

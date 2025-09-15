#!/usr/bin/env python3
"""
Clean up old adapter files and exports, keeping only experiment directories.
"""

import os
import shutil
from pathlib import Path
import argparse
from datetime import datetime

DEFAULT_ADAPTERS_PATH = "./avro-phi3-adapters"
DEFAULT_EXPORTS_PATH = "./exports"

def format_size(size_bytes):
    """Convert bytes to human readable format."""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024.0:
            if unit == 'B':
                return f"{size_bytes:.0f} {unit}"
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f} PB"

def cleanup_adapters(base_path=DEFAULT_ADAPTERS_PATH, dry_run=False):
    """Remove old adapter files and keep only experiment directories."""
    base_dir = Path(base_path)

    if not base_dir.exists():
        print(f"Directory {base_path} does not exist")
        return

    files_to_remove = []
    dirs_to_remove = []
    experiment_dirs = []

    # Identify what to keep and what to remove
    for item in base_dir.iterdir():
        if item.is_file():
            # All root-level files are old format
            files_to_remove.append(item)
        elif item.is_dir():
            # Check if it's an experiment directory (new format)
            if "-r" in item.name and "-a" in item.name and "-e" in item.name:
                experiment_dirs.append(item)
            elif item.name.startswith("checkpoint-"):
                # Old checkpoint directories
                dirs_to_remove.append(item)
            else:
                # Unknown directory - check if it has experiment metadata
                metadata_file = item / "experiment_metadata.json"
                if metadata_file.exists():
                    experiment_dirs.append(item)
                else:
                    # Likely old format
                    dirs_to_remove.append(item)

    print("\n" + "="*60)
    print("CLEANUP REPORT")
    print("="*60)

    # Report findings
    if files_to_remove:
        print(f"\n📄 Old files to remove ({len(files_to_remove)}):")
        for f in files_to_remove:
            size = f.stat().st_size
            print(f"   - {f.name} ({format_size(size)})")

    if dirs_to_remove:
        print(f"\n📁 Old directories to remove ({len(dirs_to_remove)}):")
        for d in dirs_to_remove:
            # Calculate directory size
            size = sum(f.stat().st_size for f in d.rglob('*') if f.is_file())
            print(f"   - {d.name}/ ({format_size(size)})")

    if experiment_dirs:
        print(f"\n✅ Experiments to keep ({len(experiment_dirs)}):")
        for e in sorted(experiment_dirs, key=lambda x: x.name):
            print(f"   - {e.name}/")

    if not files_to_remove and not dirs_to_remove:
        print("\n✨ Directory is already clean!")
        return

    # Calculate total space to be freed
    total_size = 0
    for f in files_to_remove:
        total_size += f.stat().st_size
    for d in dirs_to_remove:
        total_size += sum(f.stat().st_size for f in d.rglob('*') if f.is_file())

    print(f"\n💾 Total space to be freed: {format_size(total_size)}")

    if dry_run:
        print("\n🔍 DRY RUN - No files were actually removed")
        print("   Run without --dry-run to perform cleanup")
    else:
        print("\n🗑️ Removing old files...")

        # Remove files
        for f in files_to_remove:
            try:
                f.unlink()
                print(f"   ✓ Removed {f.name}")
            except Exception as e:
                print(f"   ✗ Failed to remove {f.name}: {e}")

        # Remove directories
        for d in dirs_to_remove:
            try:
                shutil.rmtree(d)
                print(f"   ✓ Removed {d.name}/")
            except Exception as e:
                print(f"   ✗ Failed to remove {d.name}/: {e}")

        print("\n✅ Cleanup complete!")

    print("="*60)

def cleanup_exports(base_path=DEFAULT_EXPORTS_PATH, dry_run=False):
    """Clean up exports directory."""
    base_dir = Path(base_path)

    if not base_dir.exists():
        print(f"Directory {base_path} does not exist")
        return

    files_to_remove = []
    dirs_to_remove = []

    # In exports, we typically want to remove everything
    # unless there's a specific pattern to keep
    for item in base_dir.iterdir():
        if item.is_file():
            files_to_remove.append(item)
        elif item.is_dir():
            dirs_to_remove.append(item)

    print("\n" + "="*60)
    print("EXPORTS CLEANUP REPORT")
    print("="*60)

    if files_to_remove:
        print(f"\n📄 Export files to remove ({len(files_to_remove)}):")
        for f in files_to_remove:
            size = f.stat().st_size
            print(f"   - {f.name} ({format_size(size)})")

    if dirs_to_remove:
        print(f"\n📁 Export directories to remove ({len(dirs_to_remove)}):")
        for d in dirs_to_remove:
            # Calculate directory size
            size = sum(f.stat().st_size for f in d.rglob('*') if f.is_file())
            print(f"   - {d.name}/ ({format_size(size)})")

    if not files_to_remove and not dirs_to_remove:
        print("\n✨ Exports directory is already clean!")
        return

    # Calculate total space to be freed
    total_size = 0
    for f in files_to_remove:
        total_size += f.stat().st_size
    for d in dirs_to_remove:
        total_size += sum(f.stat().st_size for f in d.rglob('*') if f.is_file())

    print(f"\n💾 Total space to be freed: {format_size(total_size)}")

    if dry_run:
        print("\n🔍 DRY RUN - No files were actually removed")
        print("   Run without --dry-run to perform cleanup")
    else:
        print("\n🗑️ Removing export files...")

        # Remove files
        for f in files_to_remove:
            try:
                f.unlink()
                print(f"   ✓ Removed {f.name}")
            except Exception as e:
                print(f"   ✗ Failed to remove {f.name}: {e}")

        # Remove directories
        for d in dirs_to_remove:
            try:
                shutil.rmtree(d)
                print(f"   ✓ Removed {d.name}/")
            except Exception as e:
                print(f"   ✗ Failed to remove {d.name}/: {e}")

        print("\n✅ Cleanup complete!")

    print("="*60)

def main():
    parser = argparse.ArgumentParser(description="Clean up adapters and exports")
    parser.add_argument("target", choices=["adapters", "exports", "both"],
                       help="What to clean: adapters, exports, or both")
    parser.add_argument("--adapters-path", default=DEFAULT_ADAPTERS_PATH,
                       help="Path to adapters directory")
    parser.add_argument("--exports-path", default=DEFAULT_EXPORTS_PATH,
                       help="Path to exports directory")
    parser.add_argument("--dry-run", action="store_true",
                       help="Show what would be removed without removing")

    args = parser.parse_args()

    if args.target in ["adapters", "both"]:
        cleanup_adapters(args.adapters_path, args.dry_run)

    if args.target in ["exports", "both"]:
        cleanup_exports(args.exports_path, args.dry_run)

if __name__ == "__main__":
    main()
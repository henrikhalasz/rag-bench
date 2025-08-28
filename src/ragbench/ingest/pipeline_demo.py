#!/usr/bin/env python3
"""
Complete chunking pipeline demonstration.
Shows the recommended workflow with all QC checks.
"""
import argparse
import subprocess
import sys
from pathlib import Path

def run_command(cmd, description):
    """Run a command and handle errors."""
    print(f"\n🔄 {description}")
    print(f"   Command: {' '.join(map(str, cmd))}")
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"❌ Failed: {description}")
        print(f"Error: {result.stderr}")
        return False
    
    if result.stdout.strip():
        print(f"✅ {result.stdout.strip()}")
    else:
        print(f"✅ {description} completed successfully")
    
    return True

def main():
    parser = argparse.ArgumentParser(description="Complete chunking pipeline with QC")
    parser.add_argument("--input", type=Path, required=True, help="Parser JSONL file")
    parser.add_argument("--output-dir", type=Path, required=True, help="Output directory")
    parser.add_argument("--max-tokens", type=int, default=900)
    parser.add_argument("--overlap-tokens", type=int, default=120)
    parser.add_argument("--skip-determinism", action="store_true", help="Skip determinism check (faster)")
    parser.add_argument("--skip-dedup", action="store_true", help="Skip deduplication")
    
    args = parser.parse_args()
    
    if not args.input.exists():
        print(f"❌ Input file not found: {args.input}")
        return 1
    
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"🚀 Starting chunking pipeline")
    print(f"   Input: {args.input}")
    print(f"   Output: {args.output_dir}")
    
    # Step 1: Run chunking
    chunk_cmd = [
        sys.executable, "-m", "src.ragbench.ingest.chunker",
        "--input", str(args.input),
        "--out", str(args.output_dir),
        "--max-tokens", str(args.max_tokens),
        "--overlap-tokens", str(args.overlap_tokens)
    ]
    
    if not run_command(chunk_cmd, "Running chunker"):
        return 1
    
    # Find the output files
    chunks_files = list(args.output_dir.glob("*_chunks.jsonl"))
    if not chunks_files:
        print("❌ No chunks file found in output directory")
        return 1
    
    chunks_file = chunks_files[0]
    
    # Step 2: QC check
    qc_cmd = [sys.executable, "src/ragbench/ingest/qc_check.py", str(chunks_file)]
    if not run_command(qc_cmd, "Running QC checks"):
        return 1
    
    # Step 3: Determinism check (optional)
    if not args.skip_determinism:
        det_cmd = [
            sys.executable, "src/ragbench/ingest/determinism_check.py",
            "--input", str(args.input),
            "--max-tokens", str(args.max_tokens),
            "--overlap-tokens", str(args.overlap_tokens)
        ]
        if not run_command(det_cmd, "Running determinism check"):
            return 1
    
    # Step 4: Deduplication (optional)
    if not args.skip_dedup:
        dedup_file = chunks_file.parent / f"{chunks_file.stem}_dedup.jsonl"
        dedup_cmd = [
            sys.executable, "src/ragbench/ingest/dedup_chunks.py",
            str(chunks_file), str(dedup_file), "--stats"
        ]
        if not run_command(dedup_cmd, "Running deduplication"):
            return 1
        
        # QC check on deduplicated output
        qc_dedup_cmd = [sys.executable, "src/ragbench/ingest/qc_check.py", str(dedup_file)]
        if not run_command(qc_dedup_cmd, "QC check on deduplicated chunks"):
            return 1
    
    print(f"\n🎉 Pipeline completed successfully!")
    print(f"   Final chunks: {chunks_file}")
    if not args.skip_dedup:
        print(f"   Deduplicated: {dedup_file}")
    
    # Show manifest summary
    manifest_files = list(args.output_dir.glob("*_manifest.json"))
    if manifest_files:
        print(f"   Manifest: {manifest_files[0]}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())

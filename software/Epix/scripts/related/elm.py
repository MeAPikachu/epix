#!/usr/bin/env python3
import subprocess
from pathlib import Path
from datetime import datetime

LOCAL_DIR = Path("/data/L1")
REMOTE_PREFIX = "my-elm:mossbauer-backup/L1"

# the main file that we want to save is the L1 level files; 
# Newest-first ranks to process: 100th newest through 50th newest
START_RANK = 100
END_RANK = 50

# This script needs to be run by the root because I only set it up on root ; 
# We are using rclone command; 
RCLONE_BASE_CMD = [
	"rclone",
	"--s3-no-check-bucket",
]

def log(msg: str) -> None:
	now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
	print(f"[{now}] {msg}", flush=True)

def run_cmd(cmd: list[str]) -> subprocess.CompletedProcess:
	return subprocess.run(
		cmd,
		stdout=subprocess.PIPE,
		stderr=subprocess.PIPE,
		text=True,
	)

def remote_file_exists(remote_path: str) -> bool:
	# Checks whether this exact object exists on the remote.
	cmd = RCLONE_BASE_CMD + ["lsf", remote_path]
	result = run_cmd(cmd)
	return result.returncode == 0 and result.stdout.strip() != ""

def upload_file(local_path: Path, remote_path: str) -> bool:
	cmd = RCLONE_BASE_CMD + ["copyto", str(local_path), remote_path]
	result = run_cmd(cmd)
	if result.returncode != 0:
		log(f"Upload failed for {local_path.name}")
		if result.stderr.strip():
			log(result.stderr.strip())
		return False
	return True

def get_files_sorted_newest_first(directory: Path) -> list[Path]:
	files = [p for p in directory.iterdir() if p.is_file()]
	files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
	return files

def main() -> int:
	if not LOCAL_DIR.exists():
		log(f"Local directory does not exist: {LOCAL_DIR}")
		return 1

	files = get_files_sorted_newest_first(LOCAL_DIR)
	count = len(files)

	if count < START_RANK:
		log(f"Found only {count} files in {LOCAL_DIR}; need at least {START_RANK}. Nothing to do.")
		return 0

	log(f"Found {count} files in {LOCAL_DIR}")
	log(f"Processing ranks {START_RANK} through {END_RANK} (newest first)")

	# Convert human ranks to Python slice indexes:
	# rank 1 -> index 0
	# want 10..30 inclusive, but we will iterate from 30 down to 10
	for rank in range(START_RANK, END_RANK - 1, -1):
		idx = rank - 1
		local_path = files[idx]
		remote_path = f"{REMOTE_PREFIX}/{local_path.name}"

		if remote_file_exists(remote_path):
			log(f"Exists on remote, skipping: rank {rank}: {local_path.name}")
			continue

		log(f"Missing on remote, uploading: rank {rank}: {local_path.name}")
		ok = upload_file(local_path, remote_path)
		if ok:
			log(f"Uploaded: {local_path.name}")

	return 0

if __name__ == "__main__":
	raise SystemExit(main())
#!/usr/bin/env python3
import fcntl
import subprocess
from pathlib import Path
from datetime import datetime

LOCAL_DIR = Path("/data/L1")
REMOTE_PREFIX = "my-elm:mossbauer-backup/L1"
LOCKFILE = "/tmp/elm_upload.lock"

# Process 30th newest through 10th newest
START_RANK = 100
END_RANK = 50

RCLONE_BASE_CMD = [
    "rclone",
    "--s3-no-check-bucket",
    "--multi-thread-streams", "0",
    "--retries", "1",
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
    result = run_cmd(RCLONE_BASE_CMD + ["lsf", remote_path])
    return result.returncode == 0 and result.stdout.strip() != ""

def upload_file(local_path: Path, remote_path: str) -> bool:
    size_gib = local_path.stat().st_size / (1024 ** 3)
    log(f"Uploading {local_path.name} ({size_gib:.3f} GiB)")

    result = run_cmd(RCLONE_BASE_CMD + ["copyto", str(local_path), remote_path])

    if result.returncode != 0:
        log(f"Upload failed: {local_path.name}")
        if result.stderr.strip():
            log(result.stderr.strip())
        return False

    log(f"Uploaded: {local_path.name}")
    return True

def get_files_sorted_newest_first(directory: Path) -> list[Path]:
    files = [
        p for p in directory.iterdir()
        if p.is_file() and p.name.startswith("L1_")
    ]
    files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return files

def main() -> int:
    with open(LOCKFILE, "w") as lockf:
        try:
            fcntl.flock(lockf, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            log("Another instance is already running; exiting.")
            return 0

        if not LOCAL_DIR.exists():
            log(f"Directory not found: {LOCAL_DIR}")
            return 1

        files = get_files_sorted_newest_first(LOCAL_DIR)
        count = len(files)

        if count < START_RANK:
            log(f"Only {count} matching files found; need at least {START_RANK}. Nothing to do.")
            return 0

        log(f"Found {count} files in {LOCAL_DIR}")
        log(f"Processing ranks {START_RANK} through {END_RANK} (newest first)")

        for rank in range(START_RANK, END_RANK - 1, -1):
            local_path = files[rank - 1]
            remote_path = f"{REMOTE_PREFIX}/{local_path.name}"

            if remote_file_exists(remote_path):
                log(f"Skip existing: rank {rank}: {local_path.name}")
                continue

            log(f"Missing on remote: rank {rank}: {local_path.name}")
            upload_file(local_path, remote_path)

    return 0

if __name__ == "__main__":
    raise SystemExit(main())
#!/usr/bin/env python3
"""
Job scheduler for COLMAP reconstruction and shape matching.

Two modes:
1. Streaming mode: Automatically schedules new jobs as old ones finish
2. Manual mode: Schedules first N folders and waits for completion

Resource constraints:
- 2 GPUs (IDs: 1, 2)
- Max 10 COLMAP jobs per GPU (configurable)
- 128 CPU cores total
- Dynamic CPU allocation based on active jobs
"""

import os
import sys
import json
import time
import shutil
import subprocess
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import psutil
import pynvml

# Make registration_check / quality_check (siblings of this file) importable
# regardless of CWD
_SCHEDULER_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCHEDULER_DIR))
from registration_check import (  # noqa: E402
    registration_ratio,
    is_well_registered,
    REGISTRATION_THRESHOLD,
)
from quality_check import detect_quality_warnings  # noqa: E402

# ==================== Configuration ====================

class Config:
    """Scheduler configuration"""
    GPU_IDS = [0, 1, 2, 3]
    MAX_COLMAP_PER_GPU = 1  # Maximum COLMAP jobs per GPU
    TOTAL_CPU_CORES = 128
    CPU_CORES_PER_COLMAP = 8  # CPU threads allocated per COLMAP job
    MIN_SHAPE_MATCHING_WORKERS = 8  # Minimum workers for shape matching
    MAX_CONCURRENT_SHAPE_MATCHING = 4  # Maximum concurrent shape matching jobs
    MIN_GPU_MEMORY_MB = 4096  # Minimum free GPU memory (MB) to launch COLMAP
    MAX_GPU_UTILIZATION = 80  # Maximum GPU utilization (%) to launch COLMAP
    MAX_MEMORY_UTILIZATION = 90  # Maximum memory utilization (%) to launch COLMAP
    POLL_INTERVAL_SEC = 10  # How often to check job status
    MATERIAL_SCAN_INTERVAL_SEC = 30  # How often to scan for new materials in auto mode
    STATE_FILE_NAME = "scheduler_state.json"  # State file name (saved in dataset folder)

    # COLMAP stages a local copy of ldr/ plus its outputs under COLMAP_TMP_BASE.
    # launch_colmap() exports it to the scripts as COLMAP_TMP, so the disk gate
    # below measures the filesystem they actually write to. Precedence:
    # $COLMAP_TMP, then the production loop device /mnt/colmap_tmp if it
    # exists, else the scripts' own default. Require at least MIN_FREE_DISK_GB
    # free before launching a new COLMAP job, so disk-full failures can't take
    # down the whole batch mid-flight (the scripts remove their staging dir on
    # every exit path, success or failure).
    COLMAP_TMP_BASE = os.environ.get("COLMAP_TMP") or (
        "/mnt/colmap_tmp" if os.path.isdir("/mnt/colmap_tmp") else "/tmp/robocloth_colmap")
    MIN_FREE_DISK_GB = 20.0

    # Paths
    COLMAP_SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "colmap.sh")
    COLMAP_EXHAUSTIVE_SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "colmap_exhaustive.sh")
    SHAPE_MATCHING_SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "reconstruct.py")
    
    def __init__(self, dataset_root: Optional[str] = None):
        """Initialize config with dataset-specific paths."""
        self.dataset_root = dataset_root
        if dataset_root:
            self.STATE_FILE = os.path.join(dataset_root, self.STATE_FILE_NAME)
        else:
            self.STATE_FILE = self.STATE_FILE_NAME

# ==================== GPU Monitor ====================

class GPUMonitor:
    """
    Monitor GPU status using NVIDIA Management Library (pynvml).
    Provides memory, utilization, and temperature information.
    """
    
    def __init__(self):
        """Initialize NVML."""
        try:
            pynvml.nvmlInit()
            self._initialized = True
        except Exception as e:
            print(f"Warning: Failed to initialize NVML: {e}")
            self._initialized = False
    
    def __del__(self):
        """Cleanup NVML."""
        if self._initialized:
            try:
                pynvml.nvmlShutdown()
            except:
                pass
    
    def get_gpu_info(self, gpu_id: int) -> Dict[str, any]:
        """
        Get comprehensive GPU information.
        
        Args:
            gpu_id: GPU device ID
        
        Returns:
            Dict with keys:
                - memory_free_mb: Free memory in MB
                - memory_used_mb: Used memory in MB
                - memory_total_mb: Total memory in MB
                - utilization_gpu: GPU utilization percentage (0-100)
                - utilization_memory: Memory utilization percentage (0-100)
                - temperature: GPU temperature in Celsius
                - available: Whether data was successfully retrieved
        """
        if not self._initialized:
            return {"available": False}
        
        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
            
            # Memory info
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            memory_free_mb = mem_info.free // (1024 * 1024)
            memory_used_mb = mem_info.used // (1024 * 1024)
            memory_total_mb = mem_info.total // (1024 * 1024)
            
            # Utilization rates
            util_rates = pynvml.nvmlDeviceGetUtilizationRates(handle)
            utilization_gpu = util_rates.gpu
            utilization_memory = util_rates.memory
            
            # Temperature
            temperature = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
            
            return {
                "available": True,
                "memory_free_mb": memory_free_mb,
                "memory_used_mb": memory_used_mb,
                "memory_total_mb": memory_total_mb,
                "utilization_gpu": utilization_gpu,
                "utilization_memory": utilization_memory,
                "temperature": temperature
            }
        except Exception as e:
            print(f"Error getting GPU {gpu_id} info: {e}")
            return {"available": False}
    
    def is_gpu_available(self, gpu_id: int, 
                         min_memory_mb: int = 2048,
                         max_gpu_util: int = 80,
                         max_memory_util: int = 90) -> Tuple[bool, str]:
        """
        Check if GPU is available for new job based on multiple criteria.
        
        Args:
            gpu_id: GPU device ID
            min_memory_mb: Minimum free memory required (MB)
            max_gpu_util: Maximum GPU utilization allowed (%)
            max_memory_util: Maximum memory utilization allowed (%)
        
        Returns:
            Tuple of (is_available, reason)
            - is_available: True if GPU meets all criteria
            - reason: String explaining why GPU is unavailable (empty if available)
        """
        info = self.get_gpu_info(gpu_id)
        
        if not info["available"]:
            return False, "GPU info unavailable"
        
        # Check memory
        if info["memory_free_mb"] < min_memory_mb:
            return False, f"Low memory ({info['memory_free_mb']} MB < {min_memory_mb} MB)"
        
        # Check GPU utilization
        if info["utilization_gpu"] > max_gpu_util:
            return False, f"High GPU utilization ({info['utilization_gpu']}% > {max_gpu_util}%)"
        
        # Check memory utilization
        if info["utilization_memory"] > max_memory_util:
            return False, f"High memory utilization ({info['utilization_memory']}% > {max_memory_util}%)"
        
        return True, ""

# ==================== Job Status Enum ====================

class JobStatus:
    NOT_STARTED = "NOT_STARTED"
    COLMAP_QUEUED = "COLMAP_QUEUED"
    COLMAP_RUNNING = "COLMAP_RUNNING"
    COLMAP_DONE = "COLMAP_DONE"
    SHAPE_MATCHING_RUNNING = "SHAPE_MATCHING_RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"

# ==================== Utility Functions ====================

def get_least_loaded_gpu(state: Dict, config: Config, gpu_monitor: GPUMonitor) -> Optional[int]:
    """
    Find GPU with fewest running COLMAP jobs and sufficient resources.
    Checks memory, GPU utilization, and memory utilization.
    
    Returns: GPU ID or None if all GPUs are at capacity or unavailable
    """
    gpu_loads = {gpu_id: 0 for gpu_id in config.GPU_IDS}
    
    # Count running COLMAP jobs per GPU
    for material, info in state["materials"].items():
        if info["status"] == JobStatus.COLMAP_RUNNING and "gpu" in info:
            gpu_loads[info["gpu"]] += 1
    
    # Find GPU with lowest load that has capacity and available resources
    candidates = []
    for gpu_id in config.GPU_IDS:
        if gpu_loads[gpu_id] < config.MAX_COLMAP_PER_GPU:
            is_available, reason = gpu_monitor.is_gpu_available(
                gpu_id,
                min_memory_mb=config.MIN_GPU_MEMORY_MB,
                max_gpu_util=config.MAX_GPU_UTILIZATION,
                max_memory_util=config.MAX_MEMORY_UTILIZATION
            )
            
            if is_available:
                candidates.append((gpu_id, gpu_loads[gpu_id]))
            # Optionally log why GPU is unavailable (for debugging)
            # else:
            #     print(f"GPU {gpu_id} unavailable: {reason}")
    
    if not candidates:
        return None
    
    # Return GPU with lowest load
    candidates.sort(key=lambda x: x[1])
    return candidates[0][0]

def calculate_shape_matching_workers(state: Dict, config: Config) -> int:
    """
    Calculate optimal number of workers for shape matching based on current load.
    
    Strategy:
    - Reserve CPU_CORES_PER_COLMAP * active_colmap_count for COLMAP
    - Allocate remaining to shape matching (with min/max bounds)
    """
    # Count active COLMAP jobs
    active_colmap = sum(
        1 for info in state["materials"].values()
        if info["status"] == JobStatus.COLMAP_RUNNING
    )
    
    # Count active shape matching jobs
    active_shape = sum(
        1 for info in state["materials"].values()
        if info["status"] == JobStatus.SHAPE_MATCHING_RUNNING
    )
    
    # Calculate available cores
    colmap_cores = active_colmap * config.CPU_CORES_PER_COLMAP
    
    
    # Distribute remaining cores among shape matching jobs
    available_cores = config.TOTAL_CPU_CORES - colmap_cores - 8  # 8 core buffer
    workers_per_job = max(config.MIN_SHAPE_MATCHING_WORKERS, 
                          available_cores // (active_shape + 1))
    
    # Cap at 12: beyond this, shape_matching workers saturate NFS write
    # throughput and the parent-directory i_rwsem, so extra workers slow
    # the job down instead of speeding it up. Raising back to 48 brought
    # the server to a halt.
    return min(workers_per_job, 12)

def get_available_cpu_cores(config: Config, threshold: float = 80.0) -> int:
    """
    Count CPU cores that are NOT heavily loaded based on per-core usage.
    
    A core is considered "available" if its usage is below the threshold (80% by default).
    This accurately reflects which cores can handle new work, regardless of whether
    the system load is balanced or not.
    
    Args:
        config: Configuration
        threshold: CPU usage threshold (%). Cores with usage < threshold are available.
    
    Returns: Number of cores with usage below threshold
    """
    # Get per-core CPU usage (returns list with one value per core)
    per_core_usage = psutil.cpu_percent(interval=0.1, percpu=True)
    
    # Count cores below threshold
    available_cores = sum(1 for usage in per_core_usage if usage < threshold)
    
    return available_cores

def check_cpu_capacity(state: Dict, config: Config, for_shape_matching: bool = False) -> bool:
    """
    Check if there's CPU capacity for a new job based on actual CPU usage.
    
    Args:
        state: Current state
        config: Configuration
        for_shape_matching: True if checking for shape matching, False for COLMAP
    
    Returns: True if capacity available
    """
    # Get actual available CPU cores
    available_cores = get_available_cpu_cores(config)
    
    # Count active jobs for other constraints
    active_shape = sum(
        1 for info in state["materials"].values()
        if info["status"] == JobStatus.SHAPE_MATCHING_RUNNING
    )
    
    if for_shape_matching:
        # Check if we're at max concurrent shape matching jobs
        if active_shape >= config.MAX_CONCURRENT_SHAPE_MATCHING:
            return False
        
        # Estimate cores needed for new shape matching job
        estimated_workers = calculate_shape_matching_workers(state, config)
        
        # Require at least the estimated workers + 8 core buffer
        required_cores = estimated_workers + 8
        
        if available_cores >= required_cores:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] CPU check for shape matching: {available_cores} cores available, {required_cores} required -> OK")
            return True
        else:
            return False
    else:
        # For COLMAP - buffer scales with host size (was hard-coded +40 for 128-core hosts)
        required_cores = config.CPU_CORES_PER_COLMAP + max(4, config.TOTAL_CPU_CORES // 8)
        
        if available_cores >= required_cores:
            return True
        else:
            return False

def _free_disk_gb(path: str) -> float:
    """Return free space in GB on the filesystem containing `path`.

    If `path` does not exist yet, walk up to the nearest existing ancestor so
    we still measure the correct filesystem on first startup.
    """
    probe = path
    while probe and not os.path.exists(probe):
        parent = os.path.dirname(probe)
        if parent == probe:
            break
        probe = parent
    try:
        usage = shutil.disk_usage(probe or "/")
    except OSError:
        return -1.0
    return usage.free / (1024 ** 3)


def check_disk_capacity(config: Config) -> Tuple[bool, float]:
    """
    Check whether the filesystem hosting COLMAP_TMP_BASE has at least
    MIN_FREE_DISK_GB free. COLMAP jobs copy the full `ldr/` image set to
    TMP_BASE, so starting one when space is low has caused whole-batch
    failures (tmp files are only cleaned after each job finishes).

    Returns (ok, free_gb). If `free_gb` is < 0 the filesystem is unreadable;
    treat as unavailable so we don't launch blind.
    """
    free_gb = _free_disk_gb(config.COLMAP_TMP_BASE)
    ok = free_gb >= config.MIN_FREE_DISK_GB
    return ok, free_gb


def is_process_alive(pid: int) -> bool:
    """Check if a process is still running."""
    try:
        process = psutil.Process(pid)
        return process.is_running() and process.status() != psutil.STATUS_ZOMBIE
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return False

def terminate_process(pid: Optional[int], material: str, job_type: str):
    """
    Safely terminate a process and all its children.
    
    Args:
        pid: Process ID to terminate (can be None)
        material: Material name (for logging)
        job_type: "COLMAP" or "shape matching" (for logging)
    """
    if pid is None:
        return
    
    try:
        if is_process_alive(pid):
            process = psutil.Process(pid)
            
            # Get all child processes
            children = process.children(recursive=True)
            
            # Terminate children first
            for child in children:
                try:
                    child.terminate()
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
            
            # Terminate parent
            process.terminate()
            
            # Wait up to 5 seconds for graceful shutdown
            try:
                process.wait(timeout=5)
            except psutil.TimeoutExpired:
                # Force kill if still alive
                process.kill()
                for child in children:
                    try:
                        child.kill()
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        pass
                print(f"[{datetime.now().strftime('%H:%M:%S')}] Force killed {job_type} process {pid} for material {material}")
            else:
                print(f"[{datetime.now().strftime('%H:%M:%S')}] Terminated {job_type} process {pid} for material {material}")
    except psutil.NoSuchProcess:
        # Process already dead
        pass
    except Exception as e:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Error terminating {job_type} process {pid} for material {material}: {e}")

def is_material_ready(folder_path: str) -> bool:
    """
    Check if a material folder is ready for processing.
    A material is ready if scan_log.json exists and is not empty.
    
    Args:
        folder_path: Path to material folder
    
    Returns: True if material is ready for processing
    """
    scan_log_path = os.path.join(folder_path, "scan_log.json")
    
    if not os.path.exists(scan_log_path):
        return False
    
    try:
        # Check if file is not empty
        if os.path.getsize(scan_log_path) == 0:
            return False
        
        # Try to load as JSON to verify it's valid
        with open(scan_log_path, 'r') as f:
            data = json.load(f)
            return bool(data)  # Return True if not empty dict/list
    except (json.JSONDecodeError, IOError):
        return False

# ==================== State Management ====================

def load_state(state_file: str) -> Dict:
    """Load scheduler state from JSON file."""
    if os.path.exists(state_file):
        with open(state_file, 'r') as f:
            return json.load(f)
    else:
        return {
            "materials": {},
            "last_updated": None,
            "mode": None
        }

def save_state(state: Dict, state_file: str):
    """Save scheduler state to JSON file."""
    state["last_updated"] = datetime.now().isoformat()
    with open(state_file, 'w') as f:
        json.dump(state, f, indent=2)

def fix_material_status_by_timestamps(info: Dict, material: str, verbose: bool = False) -> bool:
    """
    Fix material status based on completion timestamps (simple deterministic rules).
    
    Rules:
    1. Has shape_matching_end_time? → COMPLETED
    2. Has colmap_end_time (but no shape_matching_end_time)? → COLMAP_DONE
    3. Has neither end time but marked as RUNNING? → Reset to NOT_STARTED
    
    Args:
        info: Material info dict
        material: Material name (for logging)
        verbose: Whether to print fix messages
    
    Returns: True if status was fixed
    """
    current_status = info["status"]
    shape_end = info.get("shape_matching_end_time")
    colmap_end = info.get("colmap_end_time")
    pid = info.get("pid")
    
    # Skip if already in final states or FAILED (FAILED jobs are handled by reset_failed_jobs)
    if current_status in [JobStatus.COMPLETED, JobStatus.NOT_STARTED, JobStatus.FAILED]:
        return False
    
    # Rule 1: Has shape matching end time → COMPLETED
    if shape_end and current_status != JobStatus.COMPLETED:
        terminate_process(pid, material, "shape matching")
        info["status"] = JobStatus.COMPLETED
        info["pid"] = None
        info["gpu"] = None
        info["workers"] = None
        if verbose:
            print(f"  Fixed material {material}: {current_status} → COMPLETED (has shape_matching_end_time)")
        return True
    
    # Rule 2: Has COLMAP end time but not shape matching → COLMAP_DONE
    elif colmap_end and current_status not in [JobStatus.COLMAP_DONE, JobStatus.COMPLETED]:
        terminate_process(pid, material, "COLMAP")
        info["status"] = JobStatus.COLMAP_DONE
        info["pid"] = None
        info["gpu"] = None
        info["workers"] = None
        if verbose:
            print(f"  Fixed material {material}: {current_status} → COLMAP_DONE (has colmap_end_time)")
        return True
    
    # Rule 3: No end times but marked as RUNNING → reset to NOT_STARTED
    elif not colmap_end and current_status in [JobStatus.COLMAP_RUNNING, JobStatus.SHAPE_MATCHING_RUNNING]:
        job_type = "COLMAP" if current_status == JobStatus.COLMAP_RUNNING else "shape matching"
        terminate_process(pid, material, job_type)
        info["status"] = JobStatus.NOT_STARTED
        info["pid"] = None
        info["gpu"] = None
        info["workers"] = None
        info["colmap_start_time"] = None
        info["shape_matching_start_time"] = None
        if verbose:
            print(f"  Fixed material {material}: {current_status} → NOT_STARTED (no end times)")
        return True
    
    return False

def archive_rejected_sparse(folder_path: str, log_prefix: str = "  ") -> Optional[str]:
    """
    Move <folder>/sparse aside as sparse_seq_failed-<YYYYmmdd-HHMMSS> before an
    exhaustive COLMAP retry. Never deletes anything: earlier archives (including
    a legacy unsuffixed sparse_seq_failed/) survive, and a same-second clash gets
    a numeric suffix — mirroring the sparse.prev-<ts> dirs colmap.sh keeps.

    Returns the archive path, or None if there was no sparse/ to archive or the
    rename failed (a warning is printed; the caller proceeds either way).
    """
    sparse_dir = os.path.join(folder_path, "sparse")
    if not os.path.isdir(sparse_dir):
        return None
    stamp = datetime.now().strftime('%Y%m%d-%H%M%S')
    backup_dir = os.path.join(folder_path, f"sparse_seq_failed-{stamp}")
    n = 1
    while os.path.exists(backup_dir):
        backup_dir = os.path.join(folder_path, f"sparse_seq_failed-{stamp}-{n}")
        n += 1
    try:
        os.rename(sparse_dir, backup_dir)
    except OSError as e:
        print(f"{log_prefix}WARNING: could not archive {sparse_dir} -> {backup_dir}: {e}")
        return None
    return backup_dir


def verify_completed_materials(state: Dict, verbose: bool = True, skip_list: Optional[set] = None) -> Tuple[int, int]:
    """
    Sanity-check every COMPLETED material on scheduler restart and reset the
    bad ones so they get reprocessed automatically.

    Two failure modes are recognised:

      1. COLMAP registration ratio < REGISTRATION_THRESHOLD
         → reset to NOT_STARTED with colmap_variant="exhaustive".
         The next dispatch loop will rerun COLMAP using exhaustive_matcher.

      2. Registration is fine but observations_structured.npz is missing/empty
         → reset to COLMAP_DONE.
         The next dispatch loop will rerun shape_matching, which now writes
         observations_structured.npz inline (cheap, ~5 min, no GPU).

    Materials that pass both checks are left untouched. This replaces the
    standalone rerun_failed_colmap.py script.

    Returns: (n_reset_for_colmap, n_reset_for_shape_matching)
    """
    n_colmap_reset = 0
    n_shape_reset = 0
    skip_list = skip_list or set()

    for material, info in state["materials"].items():
        if material in skip_list:
            continue
        if info["status"] != JobStatus.COMPLETED:
            continue

        folder_path = info["folder_path"]

        # Check 1: registration health
        n_reg, n_scans, ratio = registration_ratio(folder_path)
        if ratio < 0:
            if verbose:
                print(f"  Sanity check material {material}: cannot read sparse/scan_log "
                      f"(n_reg={n_reg}, K={n_scans}) → reset for exhaustive COLMAP")
            info["status"] = JobStatus.NOT_STARTED
            info["colmap_variant"] = "exhaustive"
            info["colmap_start_time"] = None
            info["colmap_end_time"] = None
            info["shape_matching_start_time"] = None
            info["shape_matching_end_time"] = None
            info["pid"] = None
            info["gpu"] = None
            info["workers"] = None
            info["error"] = None
            info["ready"] = is_material_ready(folder_path)
            n_colmap_reset += 1
            continue

        if ratio < REGISTRATION_THRESHOLD:
            if verbose:
                print(f"  Sanity check material {material}: registration {n_reg}/{n_scans} "
                      f"({ratio*100:.1f}%) below {REGISTRATION_THRESHOLD*100:.0f}% "
                      f"→ reset for exhaustive COLMAP")
            # Archive the bad sparse (timestamped, never deleted) so the
            # exhaustive run starts clean
            archive_rejected_sparse(folder_path, log_prefix="    ")
            info["status"] = JobStatus.NOT_STARTED
            info["colmap_variant"] = "exhaustive"
            info["colmap_start_time"] = None
            info["colmap_end_time"] = None
            info["shape_matching_start_time"] = None
            info["shape_matching_end_time"] = None
            info["pid"] = None
            info["gpu"] = None
            info["workers"] = None
            info["error"] = None
            info["ready"] = is_material_ready(folder_path)
            n_colmap_reset += 1
            continue

        # Check 2: structured observations exists and is non-empty
        structured = os.path.join(folder_path, "observations_structured.npz")
        if not os.path.exists(structured) or os.path.getsize(structured) == 0:
            if verbose:
                print(f"  Sanity check material {material}: registration OK "
                      f"({n_reg}/{n_scans}, {ratio*100:.1f}%) but observations_structured.npz "
                      f"missing → reset to COLMAP_DONE for shape_matching rerun")
            info["status"] = JobStatus.COLMAP_DONE
            info["shape_matching_start_time"] = None
            info["shape_matching_end_time"] = None
            info["pid"] = None
            info["workers"] = None
            info["error"] = None
            n_shape_reset += 1
            continue

        # Check 3 (advisory): re-scan quality warnings. Always refreshes so
        # threshold changes or log re-writes are picked up. Silent here to
        # avoid spamming the log on restart with potentially hundreds of
        # warnings; use `show_status` to see them after startup.
        _record_quality_warnings(material, info, announce=False)

    return n_colmap_reset, n_shape_reset


def reset_failed_jobs(state: Dict, verbose: bool = True, force_restart_colmap: bool = True, skip_list: Optional[set] = None) -> int:
    """
    Reset failed jobs (and COLMAP_DONE jobs if force_restart_colmap=True) to retry.
    - If force_restart_colmap=True (default): Reset FAILED and COLMAP_DONE to NOT_STARTED to retry COLMAP
    - If force_restart_colmap=False: 
        - If COLMAP failed → reset to NOT_STARTED
        - If shape matching failed (COLMAP completed) → reset to COLMAP_DONE
    
    Args:
        state: Current scheduler state
        verbose: Whether to print reset messages
        force_restart_colmap: If True, always restart from COLMAP regardless of previous progress
    
    Returns: Number of jobs reset
    """
    reset_count = 0
    skip_list = skip_list or set()
    for material, info in state["materials"].items():
        if material in skip_list:
            continue
        # Determine which statuses to reset
        should_reset = info["status"] == JobStatus.FAILED
        
        # Also reset COLMAP_DONE jobs if force_restart_colmap is True
        if force_restart_colmap and info["status"] == JobStatus.COLMAP_DONE:
            should_reset = True
        
        if should_reset:
            folder_path = info["folder_path"]
            
            # Determine which stage failed by checking if COLMAP completed
            colmap_completed = (info.get("colmap_end_time") is not None and 
                              check_colmap_completion(folder_path))
            
            if colmap_completed and not force_restart_colmap:
                # COLMAP succeeded, shape matching failed
                # Reset to COLMAP_DONE so shape matching will be retried
                info["status"] = JobStatus.COLMAP_DONE
                info["pid"] = None
                info["workers"] = None
                info["error"] = None
                # Keep colmap_start_time and colmap_end_time
                # Keep gpu as None (already released)
                
                reset_count += 1
                if verbose:
                    print(f"  Reset material {material}: COLMAP OK, will retry shape matching")
            else:
                # COLMAP failed or didn't complete, or force_restart_colmap is True
                # Reset to NOT_STARTED to retry from beginning
                info["status"] = JobStatus.NOT_STARTED
                info["pid"] = None
                info["gpu"] = None
                info["workers"] = None
                info["colmap_start_time"] = None
                info["colmap_end_time"] = None
                info["shape_matching_start_time"] = None
                info["shape_matching_end_time"] = None
                info["error"] = None
                
                # Update ready status
                info["ready"] = is_material_ready(folder_path)
                
                reset_count += 1
                if verbose:
                    ready_status = "ready" if info["ready"] else "not ready (no scan_log.json)"
                    print(f"  Reset material {material}: will retry COLMAP ({ready_status})")
    
    return reset_count

def force_redo_materials(state: Dict, material_ids: List[str], verbose: bool = True) -> int:
    """
    Force-reset specific materials to NOT_STARTED so COLMAP and shape matching
    are re-run from scratch.  Works regardless of current status (COMPLETED,
    COLMAP_DONE, RUNNING, FAILED, etc.).  Running processes are NOT killed here
    — the scheduler's normal update loop will notice the status change.

    Args:
        state: Current scheduler state
        material_ids: List of material ID strings to reset (folder names)
        verbose: Whether to print reset messages

    Returns: Number of materials actually reset
    """
    reset_count = 0
    for mid in material_ids:
        if mid not in state["materials"]:
            if verbose:
                print(f"  Warning: material '{mid}' not found in state — skipping")
            continue

        info = state["materials"][mid]
        old_status = info["status"]

        # Kill running process if any
        if info.get("pid") is not None:
            terminate_process(info["pid"], mid,
                              "COLMAP" if old_status == JobStatus.COLMAP_RUNNING else "shape matching")

        info["status"] = JobStatus.NOT_STARTED
        info["pid"] = None
        info["gpu"] = None
        info["workers"] = None
        info["colmap_start_time"] = None
        info["colmap_end_time"] = None
        info["shape_matching_start_time"] = None
        info["shape_matching_end_time"] = None
        info["error"] = None
        info["colmap_variant"] = "sequential"
        info["colmap_attempts"] = 0
        info["ready"] = is_material_ready(info["folder_path"])

        reset_count += 1
        if verbose:
            ready_tag = "ready" if info["ready"] else "not ready (no scan_log.json)"
            print(f"  Force-redo material {mid}: {old_status} -> NOT_STARTED ({ready_tag})")

    return reset_count


def load_skip_list(dataset_root: str) -> set:
    """
    Load material IDs to skip from skip.txt file in dataset folder.
    
    Args:
        dataset_root: Path to dataset root folder
    
    Returns:
        Set of material IDs (as strings) to skip
    """
    skip_file = os.path.join(dataset_root, "skip.txt")
    skip_list = set()
    
    if os.path.exists(skip_file):
        try:
            with open(skip_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#'):  # Skip empty lines and comments
                        try:
                            # Parse comma-separated values or single value
                            ids = [id.strip() for id in line.split(',')]
                            skip_list.update(ids)
                        except ValueError:
                            print(f"Warning: Invalid material ID in skip.txt: {line}")
            print(f"Loaded skip list from {skip_file}: {sorted(skip_list, key=lambda x: int(x) if x.isdigit() else 0)}")
        except Exception as e:
            print(f"Warning: Failed to read skip.txt: {e}")
    
    return skip_list

def initialize_materials(dataset_root: str, state: Dict, verbose: bool = True, reset_failed: bool = False, fix_existing: bool = True, force_restart_colmap: bool = True, only: Optional[List[str]] = None) -> Tuple[List[str], int]:
    """
    Scan dataset folder for material subfolders (0, 1, 2, ..., 10, ..., 100, ...).
    Initialize state for new materials.

    Args:
        dataset_root: Path to dataset root folder
        state: Current scheduler state
        verbose: Whether to print discovery messages
        reset_failed: Whether to reset failed jobs to NOT_STARTED
        fix_existing: Whether to fix inconsistent states for existing materials (should only be done once on startup)
        force_restart_colmap: If True, always restart failed jobs from COLMAP (default True)
        only: If set, restrict ALL operations (discovery, verify, reset, dispatch) to
              this whitelist of material IDs. Overrides skip.txt for whitelisted IDs.
              Other materials' state is left completely untouched.

    Returns:
        Tuple of (sorted list of material folder names, count of new materials found)
    """
    dataset_path = Path(dataset_root)
    if not dataset_path.exists():
        raise ValueError(f"Dataset root does not exist: {dataset_root}")

    # Load skip list
    skip_list = load_skip_list(dataset_root)

    # If --only is set, treat all non-whitelisted materials (in state OR on disk)
    # as if they were in skip.txt. Whitelisted IDs override any skip.txt entry.
    only_set = set(str(m) for m in only) if only else None
    if only_set is not None:
        skip_list = (skip_list - only_set) | (set(state["materials"].keys()) - only_set)
        if verbose:
            print(f"--only restricting to {sorted(only_set, key=lambda x: int(x) if x.isdigit() else 0)} "
                  f"(other materials' state untouched)")

    # Find all numeric subfolders
    material_folders = []
    skipped_folders = []
    for item in dataset_path.iterdir():
        if item.is_dir() and item.name.isdigit():
            if item.name in skip_list or (only_set is not None and item.name not in only_set):
                skipped_folders.append(item.name)
            else:
                material_folders.append(item.name)
    
    if skipped_folders and verbose:
        if len(skipped_folders) > 30:
            print(f"Skipping {len(skipped_folders)} material(s) (skip.txt + --only filter)")
        else:
            print(f"Skipping {len(skipped_folders)} material(s) from skip.txt: {sorted(skipped_folders, key=int)}")
    
    # Sort numerically
    material_folders.sort(key=int)
    
    # Process materials: initialize new ones and fix existing ones
    new_count = 0
    fixed_count = 0
    
    for material in material_folders:
        if material not in state["materials"]:
            # NEW MATERIAL: Initialize
            folder_path = str(dataset_path / material)
            state["materials"][material] = {
                "status": JobStatus.NOT_STARTED,
                "folder_path": folder_path,
                "pid": None,
                "gpu": None,
                "workers": None,
                "colmap_start_time": None,
                "colmap_end_time": None,
                "shape_matching_start_time": None,
                "shape_matching_end_time": None,
                "error": None,
                "ready": is_material_ready(folder_path),
                # COLMAP variant: "sequential" (default, fast) or "exhaustive" (slow fallback).
                # Sequential is tried first; if registration < REGISTRATION_THRESHOLD,
                # the scheduler resets to NOT_STARTED with variant="exhaustive".
                "colmap_variant": "sequential",
                "colmap_attempts": 0,
            }
            new_count += 1
            if verbose:
                ready_status = "ready" if state["materials"][material]["ready"] else "not ready (no scan_log.json)"
                print(f"  New material {material}: {ready_status}")
        else:
            # EXISTING MATERIAL: Fix status based on timestamps (only if fix_existing=True)
            info = state["materials"][material]
            folder_path = info["folder_path"]

            # Backfill colmap_variant/colmap_attempts for state files written before
            # the exhaustive-retry feature existed.
            info.setdefault("colmap_variant", "sequential")
            info.setdefault("colmap_attempts", 0)

            # Update ready status
            old_ready = info.get("ready", False)
            new_ready = is_material_ready(folder_path)
            info["ready"] = new_ready
            
            # Fix status based on timestamps (simple deterministic rules) - only on startup
            if fix_existing:
                if fix_material_status_by_timestamps(info, material, verbose):
                    fixed_count += 1
            
            # Notify if material became ready
            if not old_ready and new_ready and verbose and info["status"] == JobStatus.NOT_STARTED:
                print(f"  Material {material} is now ready (scan_log.json detected)")
    
    # Sanity-check COMPLETED materials: low registration → exhaustive retry,
    # missing observations_structured.npz → re-run shape matching only.
    # Only runs when fix_existing=True (i.e. on scheduler startup), not on
    # every iteration of the streaming loop.
    n_colmap_reset = 0
    n_shape_reset = 0
    if fix_existing:
        n_colmap_reset, n_shape_reset = verify_completed_materials(state, verbose=verbose, skip_list=skip_list)

    # Reset failed jobs if requested (after fixing timestamps and verifying)
    if reset_failed:
        reset_count = reset_failed_jobs(state, verbose, force_restart_colmap=force_restart_colmap, skip_list=skip_list)
        if reset_count > 0 and verbose:
            print(f"Reset {reset_count} failed job(s) to retry\n")

    if verbose:
        if new_count > 0:
            print(f"Found {new_count} new material(s)")
        if fixed_count > 0:
            print(f"Fixed {fixed_count} inconsistent state(s) based on timestamps")
        if n_colmap_reset > 0:
            print(f"Sanity check: reset {n_colmap_reset} material(s) for exhaustive COLMAP retry")
        if n_shape_reset > 0:
            print(f"Sanity check: reset {n_shape_reset} material(s) for shape_matching rerun")
        print(f"Total materials tracked: {len(material_folders)}")

    return material_folders, new_count

# ==================== Job Launchers ====================

# Popen handles of every job launched by *this* scheduler process, keyed by
# pid, so update_job_status() can read the real exit status back instead of
# inferring success from log lines alone. Jobs inherited from a previous
# scheduler process (via the state file) have no handle here — see
# child_exit_status().
_CHILD_PROCESSES: Dict[int, subprocess.Popen] = {}


def spawn_logged_job(cmd: List[str], log_file: str, env: Optional[Dict[str, str]] = None) -> subprocess.Popen:
    """
    Launch `cmd` detached (own session) with stdout+stderr redirected to
    `log_file` and remember the Popen handle for child_exit_status().

    `cmd` must be an argv list — never a shell string — so material paths are
    passed verbatim without shell interpolation.
    """
    if isinstance(cmd, (str, bytes)) or not all(isinstance(c, str) for c in cmd):
        raise TypeError(f"cmd must be a list of str, got {cmd!r}")
    with open(log_file, 'w') as f:
        process = subprocess.Popen(
            list(cmd),
            env=env,
            stdout=f,
            stderr=subprocess.STDOUT,
            start_new_session=True  # Detach from parent
        )
    _CHILD_PROCESSES[process.pid] = process
    return process


def child_exit_status(pid: Optional[int]) -> Optional[int]:
    """
    Exit status of a job launched by spawn_logged_job(): None while it is
    still running or when it was launched by a previous scheduler process
    (unknown); otherwise the status (negative = killed by that signal).
    Polling also reaps the zombie, so is_process_alive() then reports it dead.
    """
    if pid is None:
        return None
    process = _CHILD_PROCESSES.get(pid)
    if process is None:
        return None
    return process.poll()


def launch_colmap(material: str, folder_path: str, gpu_id: int, state: Dict, config: Config) -> Optional[int]:
    """
    Launch COLMAP reconstruction for a material.
    Picks the matcher script based on state["materials"][material]["colmap_variant"]:
      - "sequential" (default): colmap.sh
      - "exhaustive": colmap_exhaustive.sh
    Returns: Process PID or None on failure
    """
    try:
        info = state["materials"][material]
        variant = info.get("colmap_variant", "sequential")
        if variant == "exhaustive":
            script = config.COLMAP_EXHAUSTIVE_SCRIPT
        else:
            script = config.COLMAP_SCRIPT

        # Set environment with CPU limits
        # Note: NOT setting CUDA_VISIBLE_DEVICES here - the COLMAP script handles it internally
        env = os.environ.copy()
        env["OMP_NUM_THREADS"] = str(config.CPU_CORES_PER_COLMAP)
        env["MKL_NUM_THREADS"] = str(config.CPU_CORES_PER_COLMAP)
        # Staging base for the script — the same filesystem check_disk_capacity()
        # measures (see Config.COLMAP_TMP_BASE).
        env["COLMAP_TMP"] = config.COLMAP_TMP_BASE

        # Launch COLMAP script with physical GPU ID
        # The script will set CUDA_VISIBLE_DEVICES for each colmap command
        cmd = ["bash", script, folder_path, str(gpu_id)]

        # Redirect output to log file
        log_file = os.path.join(folder_path, "colmap.log")
        process = spawn_logged_job(cmd, log_file, env=env)

        print(f"[{datetime.now().strftime('%H:%M:%S')}] Launched COLMAP ({variant}) for material {material} on GPU {gpu_id} (PID: {process.pid})")

        # Update state
        info["status"] = JobStatus.COLMAP_RUNNING
        info["pid"] = process.pid
        info["gpu"] = gpu_id
        info["colmap_start_time"] = datetime.now().isoformat()
        info["colmap_attempts"] = info.get("colmap_attempts", 0) + 1

        return process.pid

    except Exception as e:
        print(f"Error launching COLMAP for material {material}: {e}")
        state["materials"][material]["status"] = JobStatus.FAILED
        state["materials"][material]["error"] = str(e)
        return None

def launch_shape_matching(material: str, folder_path: str, num_workers: int, state: Dict, config: Config) -> Optional[int]:
    """
    Launch shape matching for a material.
    Returns: Process PID or None on failure
    """
    try:
        # Build command with Hydra overrides
        dataset_root = os.path.dirname(os.path.normpath(folder_path))
        cmd = [
            "python", config.SHAPE_MATCHING_SCRIPT,
            f"shape_matching.folder_path={folder_path}",
            f"shape_matching.num_workers={num_workers}",
            "shape_matching.z_outlier_percentile=5.0",
            f"dataset_root={dataset_root}",
        ]
        
        # Redirect output to log file
        log_file = os.path.join(folder_path, "shape_matching.log")
        process = spawn_logged_job(cmd, log_file)
        
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Launched shape matching for material {material} with {num_workers} workers (PID: {process.pid})")
        
        # Update state
        state["materials"][material]["status"] = JobStatus.SHAPE_MATCHING_RUNNING
        state["materials"][material]["pid"] = process.pid
        state["materials"][material]["workers"] = num_workers
        state["materials"][material]["shape_matching_start_time"] = datetime.now().isoformat()
        
        return process.pid
        
    except Exception as e:
        print(f"Error launching shape matching for material {material}: {e}")
        state["materials"][material]["status"] = JobStatus.FAILED
        state["materials"][material]["error"] = str(e)
        return None

# ==================== Job Monitoring ====================

# Marker lines written by colmap.sh / colmap_exhaustive.sh at the end of colmap.log
COLMAP_FINISHED_MARKER = "== Finished COLMAP reconstruction =="
COLMAP_FAILED_MARKER = "== FAILED:"  # "== FAILED: stage=<name> exit=<code> ==" (EXIT trap)


def _log_tail(log_file: str, n: int = 20) -> List[str]:
    """Last n lines of a log file ([] if missing/unreadable)."""
    if not os.path.exists(log_file):
        return []
    try:
        with open(log_file, 'r', errors='ignore') as f:
            return f.readlines()[-n:]
    except (IOError, OSError):
        return []


def read_colmap_failure(folder_path: str) -> Optional[str]:
    """
    Return the "== FAILED: stage=<name> exit=<code> ==" line the COLMAP
    scripts' EXIT trap writes to colmap.log, or None if absent.
    """
    log_file = os.path.join(folder_path, "colmap.log")
    for line in reversed(_log_tail(log_file)):
        if COLMAP_FAILED_MARKER in line:
            return line.strip()
    return None


def _output_newer_than(path: str, start_time_iso: Optional[str]) -> bool:
    """True if `path` exists and was modified after the ISO timestamp (or if no timestamp is known)."""
    if not os.path.exists(path):
        return False
    if not start_time_iso:
        return True
    try:
        start = datetime.fromisoformat(start_time_iso).timestamp()
    except (TypeError, ValueError):
        return True
    return os.path.getmtime(path) >= start - 1.0  # 1 s slack for coarse mtimes


def _marker_exit_code(marker_line: Optional[str]) -> Optional[int]:
    """The N of a '== FAILED: stage=<s> exit=N ==' marker line (None if absent/unparseable)."""
    if not marker_line or "exit=" not in marker_line:
        return None
    try:
        return int(marker_line.split("exit=", 1)[1].split()[0])
    except ValueError:
        return None


def _signal_of(status: Optional[int]) -> Optional[int]:
    """
    Signal number behind an exit status, or None for a normal exit: Popen's
    negative status (the script itself was killed), or the 128+N status the
    scripts report when their INT/TERM trap fired or a COLMAP stage was killed
    by signal N (`set -e` propagates the child's 128+N).
    """
    if status is None:
        return None
    if status < 0:
        return -status
    if status > 128:
        return status - 128
    return None


def classify_colmap_exit(folder_path: str, returncode: Optional[int],
                         start_time_iso: Optional[str] = None) -> Tuple[str, str]:
    """
    Decide how a COLMAP run that is no longer running ended. Fail-closed: the
    run only counts as successful on an explicit success signal.

    `returncode` is child_exit_status(pid) — None when unknown (job launched
    by a previous scheduler process). Returns (outcome, reason), outcome one of
      "ok"       – colmap.sh published a validated model: exit 0 or the finished
                   marker, or (status unknown, no markers) sparse/points3D.ply
                   written after this run started
      "failed"   – the script itself failed closed (nonzero exit / FAILED
                   marker); nothing was published, sparse/ is as it was
      "vanished" – no verdict on the material: killed by a signal (negative
                   status, or the 128+N status / "exit=1xx" marker the script's
                   INT/TERM trap reports), or died without any marker. Such a
                   run is not retried with the exhaustive matcher.
    """
    failure = read_colmap_failure(folder_path)
    status = returncode if returncode is not None else _marker_exit_code(failure)
    sig = _signal_of(status)
    if sig is not None:
        reason = f"killed by signal {sig}"
        if status > 0:  # 128+N reported by the script, so its marker names the stage
            reason += f" (exit status {status})"
            if failure:
                reason += f" — {failure}"
        return "vanished", reason
    if returncode is not None and returncode != 0:
        return "failed", failure or f"COLMAP script exited with status {returncode}"
    if failure:
        return "failed", failure
    if check_colmap_completion(folder_path):
        return "ok", "completion marker in colmap.log"
    if returncode == 0:
        return "ok", "exit status 0"
    # Exit status unknown and no marker: accept the published output only if
    # this run wrote it (a fail-closed run leaves any older sparse/ in place).
    ply = os.path.join(folder_path, "sparse", "points3D.ply")
    if _output_newer_than(ply, start_time_iso):
        return "ok", "detected via output file"
    return "vanished", "process died without completion"


def check_colmap_completion(folder_path: str) -> bool:
    """
    Check if COLMAP has completed by looking for completion message in log file.
    Returns: True if completed successfully
    """
    log_file = os.path.join(folder_path, "colmap.log")
    if not os.path.exists(log_file):
        return False
    
    try:
        with open(log_file, 'r') as f:
            # Read last few lines to check for completion message
            lines = f.readlines()
            for line in reversed(lines[-20:]):  # Check last 20 lines
                if "== Finished COLMAP reconstruction ==" in line:
                    return True
    except (IOError, OSError):
        pass
    
    return False

def check_shape_matching_completion(folder_path: str) -> bool:
    """
    Check if shape matching has completed by looking for completion message in log file.
    Returns: True if completed successfully
    """
    log_file = os.path.join(folder_path, "shape_matching.log")
    if not os.path.exists(log_file):
        return False
    
    try:
        with open(log_file, 'r') as f:
            # Read last few lines to check for completion message
            lines = f.readlines()
            for line in reversed(lines[-20:]):  # Check last 20 lines
                if "Finished shape matching for" in line:
                    return True
    except (IOError, OSError):
        pass
    
    return False

def _record_quality_warnings(material: str, info: Dict, announce: bool = True) -> None:
    """
    Run post-hoc quality checks on a just-completed material and record any
    warnings in info["warnings"] (list[dict]). When announce=True, prints each
    warning so it is visible in the scheduler log alongside normal status
    output (used when a material first completes). When announce=False the
    warnings are only stored (used during bulk startup backfill).

    Safe to re-call (overwrites previous warnings list).
    """
    folder_path = info["folder_path"]
    try:
        warnings = detect_quality_warnings(folder_path)
    except Exception as e:
        # Never let the warning scanner crash the scheduler loop.
        print(f"[{datetime.now().strftime('%H:%M:%S')}] WARNING check failed "
              f"for material {material}: {e}")
        info["warnings"] = []
        return

    info["warnings"] = warnings
    if announce and warnings:
        ts = datetime.now().strftime('%H:%M:%S')
        for w in warnings:
            print(f"[{ts}] WARNING material {material} [{w['code']}]: {w['msg']}")


def evaluate_colmap_registration(material: str, info: Dict) -> Tuple[bool, str]:
    """
    Decide what to do with a COLMAP run that just finished.

    Returns (should_proceed, message):
      - (True,  msg)  : registration is healthy, transition to COLMAP_DONE
      - (False, msg)  : registration is too low; caller should reset to NOT_STARTED
                        with colmap_variant="exhaustive" (sequential→exhaustive retry),
                        OR mark FAILED if exhaustive already happened.

    Side-effect: on the first (sequential) failure the rejected sparse/ dir is
    moved aside to sparse_seq_failed-<timestamp> (archive_rejected_sparse) so
    the exhaustive retry starts clean. Nothing is ever deleted. The wrapper
    scripts now fail closed themselves (a below-threshold model is never
    published), so this path is normally only reached for a sparse/ that was
    published under a different COLMAP_REGISTRATION_THRESHOLD.
    """
    folder_path = info["folder_path"]
    n_reg, n_scans, ratio = registration_ratio(folder_path)

    if ratio < 0:
        # We can't even compute the ratio (missing images.bin or scan_log.json).
        # Treat as a hard failure of the COLMAP run.
        return False, f"could not read sparse/scan_log (n_reg={n_reg}, K={n_scans})"

    if ratio >= REGISTRATION_THRESHOLD:
        return True, f"registered {n_reg}/{n_scans} ({ratio*100:.1f}%) — healthy"

    # Unhealthy registration.
    variant = info.get("colmap_variant", "sequential")
    msg = f"registered {n_reg}/{n_scans} ({ratio*100:.1f}%) below threshold {REGISTRATION_THRESHOLD*100:.0f}%"

    if variant == "sequential":
        # Archive the rejected sparse dir (timestamped, never deleted) so the
        # exhaustive retry won't see stale data.
        archive_rejected_sparse(folder_path, log_prefix="  ")
        return False, msg + " — will retry with exhaustive matcher"

    # Already tried exhaustive and still bad — give up.
    return False, msg + f" — exhaustive retry also failed (variant={variant})"


def update_job_status(state: Dict, config: Config):
    """
    Check status of running jobs and update state.
    Uses log file completion messages to detect successful completion.
    Transitions COLMAP_DONE to SHAPE_MATCHING immediately if capacity allows.
    """
    for material, info in state["materials"].items():
        if info["status"] == JobStatus.COLMAP_RUNNING:
            folder_path = info["folder_path"]
            pid = info.get("pid")
            process_alive = pid and is_process_alive(pid)
            
            # Check for completion flag in log file
            colmap_completed = check_colmap_completion(folder_path)
            
            # If completed OR process is dead, check final status
            if colmap_completed or not process_alive:
                # The exit status is known for children of this scheduler
                # process; otherwise classify_colmap_exit() falls back to the
                # markers the COLMAP scripts write to colmap.log.
                returncode = child_exit_status(pid)
                outcome, reason = classify_colmap_exit(folder_path, returncode, info.get("colmap_start_time"))
                ts = datetime.now().strftime('%H:%M:%S')
                variant = info.get("colmap_variant", "sequential")
                terminate_process(pid, material, "COLMAP")
                info["pid"] = None
                info["gpu"] = None

                if outcome == "ok":
                    print(f"[{ts}] COLMAP completed for material {material} ({reason})")
                    info["colmap_end_time"] = datetime.now().isoformat()

                    # Registration health gate: low registration triggers an
                    # exhaustive retry (sequential variant) or hard failure
                    # (already-exhaustive variant).
                    healthy, reason = evaluate_colmap_registration(material, info)
                    if healthy:
                        print(f"  → {reason}")
                        info["status"] = JobStatus.COLMAP_DONE
                    else:
                        if variant == "sequential":
                            print(f"  → {reason}")
                            print(f"  → resetting material {material} for exhaustive COLMAP retry")
                            info["colmap_variant"] = "exhaustive"
                            info["status"] = JobStatus.NOT_STARTED
                            info["colmap_start_time"] = None
                            info["colmap_end_time"] = None
                            info["error"] = None
                        else:
                            print(f"  → {reason}")
                            info["status"] = JobStatus.FAILED
                            info["error"] = f"Low COLMAP registration after exhaustive retry: {reason}"
                elif outcome == "failed" and variant == "sequential":
                    # colmap.sh failed closed (a COLMAP stage errored or the
                    # staged model failed validation) and published nothing, so
                    # sparse/ is as it was. Same policy as a low-registration
                    # result: one exhaustive retry.
                    print(f"[{ts}] COLMAP failed for material {material}: {reason}")
                    print(f"  → resetting material {material} for exhaustive COLMAP retry")
                    info["colmap_variant"] = "exhaustive"
                    info["status"] = JobStatus.NOT_STARTED
                    info["colmap_start_time"] = None
                    info["colmap_end_time"] = None
                    info["error"] = None
                elif outcome == "failed":
                    print(f"[{ts}] COLMAP failed for material {material}: {reason}")
                    info["status"] = JobStatus.FAILED
                    info["error"] = f"COLMAP failed after exhaustive retry: {reason}"
                else:  # "vanished" — no verdict from the script
                    print(f"[{ts}] COLMAP failed for material {material} ({reason})")
                    info["status"] = JobStatus.FAILED
                    info["error"] = f"COLMAP {reason}"
        
        elif info["status"] == JobStatus.SHAPE_MATCHING_RUNNING:
            folder_path = info["folder_path"]
            pid = info.get("pid")
            process_alive = pid and is_process_alive(pid)

            # Check for completion flag in log file
            shape_completed = check_shape_matching_completion(folder_path)

            # If completed OR process is dead, check final status
            if shape_completed or not process_alive:
                returncode = child_exit_status(pid)
                if returncode is not None and returncode != 0:
                    # Fail closed: a known nonzero exit status wins over the log
                    # marker and over partial output.
                    print(f"[{datetime.now().strftime('%H:%M:%S')}] Shape matching failed for material {material} (reconstruct.py exited with status {returncode})")
                    terminate_process(pid, material, "shape matching")
                    info["status"] = JobStatus.FAILED
                    info["error"] = f"Shape matching exited with status {returncode}"
                    info["pid"] = None
                    info["workers"] = None
                elif shape_completed:
                    print(f"[{datetime.now().strftime('%H:%M:%S')}] Shape matching completed for material {material}")
                    terminate_process(pid, material, "shape matching")
                    info["status"] = JobStatus.COMPLETED
                    info["shape_matching_end_time"] = datetime.now().isoformat()
                    info["pid"] = None
                    info["workers"] = None
                    _record_quality_warnings(material, info)
                else:
                    # Process died but no completion flag - check if output exists
                    obs_folder = os.path.join(folder_path, "sparse", "observations")
                    if os.path.exists(obs_folder) and len(os.listdir(obs_folder)) > 0:
                        print(f"[{datetime.now().strftime('%H:%M:%S')}] Shape matching completed for material {material} (detected via output folder)")
                        terminate_process(pid, material, "shape matching")
                        info["status"] = JobStatus.COMPLETED
                        info["shape_matching_end_time"] = datetime.now().isoformat()
                        info["pid"] = None
                        info["workers"] = None
                        _record_quality_warnings(material, info)
                    else:
                        print(f"[{datetime.now().strftime('%H:%M:%S')}] Shape matching failed for material {material} (process died, no completion flag)")
                        terminate_process(pid, material, "shape matching")
                        info["status"] = JobStatus.FAILED
                        info["error"] = "Shape matching process died without completion"
                        info["pid"] = None
                        info["workers"] = None

def try_launch_shape_matching_jobs(state: Dict, config: Config):
    """
    Try to launch shape matching for materials that have COLMAP_DONE status.
    Prioritizes immediate launch after COLMAP completion.
    """
    for material, info in state["materials"].items():
        if info["status"] == JobStatus.COLMAP_DONE:
            # Check if we have CPU capacity
            if check_cpu_capacity(state, config, for_shape_matching=True):
                num_workers = calculate_shape_matching_workers(state, config)
                pid = launch_shape_matching(material, info["folder_path"], num_workers, state, config)
                if pid:
                    save_state(state, config.STATE_FILE)
            else:
                print(f"[{datetime.now().strftime('%H:%M:%S')}] Waiting for CPU capacity to launch shape matching for material {material}")
                break  # Wait for next iteration

# ==================== Scheduler Modes ====================

def streaming_mode(dataset_root: str, config: Config, auto_detect: bool = False, retry_failed: bool = True, force_restart_colmap: bool = True, force_redo_ids: Optional[List[str]] = None, only: Optional[List[str]] = None):
    """
    Streaming mode: Continuously schedule jobs as capacity becomes available.
    Runs until all materials are completed.

    Args:
        dataset_root: Path to dataset root folder
        config: Scheduler configuration
        auto_detect: If True, periodically scan for new materials and process them automatically
        retry_failed: If True, reset failed jobs to retry them on startup
        force_restart_colmap: If True, always restart failed jobs from COLMAP (default True)
        force_redo_ids: If provided, force these material IDs back to NOT_STARTED
    """
    print("\n" + "="*60)
    if auto_detect:
        print("STREAMING MODE: Automatic job scheduling with auto-detection")
    else:
        print("STREAMING MODE: Automatic job scheduling")
    if retry_failed:
        print(f"Failed jobs will be retried (restart from {'COLMAP' if force_restart_colmap else 'failed stage'})")
    print("="*60 + "\n")
    
    # Initialize GPU monitor
    gpu_monitor = GPUMonitor()
    
    state = load_state(config.STATE_FILE)
    state["mode"] = "streaming_auto" if auto_detect else "streaming"
    
    # Initialize materials (with failed job reset if requested)
    materials, new_count = initialize_materials(dataset_root, state, reset_failed=retry_failed, force_restart_colmap=force_restart_colmap, only=only)

    # Force-redo specific materials if requested
    if force_redo_ids:
        print(f"\nForce-redo requested for {len(force_redo_ids)} material(s):")
        n_redo = force_redo_materials(state, force_redo_ids)
        print(f"Force-reset {n_redo} material(s)\n")

    save_state(state, config.STATE_FILE)

    if not materials:
        print("No material folders found!")
        if not auto_detect:
            return
        else:
            print("Waiting for new materials...")

    print(f"Monitoring {len(materials)} materials...")
    print(f"GPU IDs: {config.GPU_IDS}")
    print(f"Max COLMAP per GPU: {config.MAX_COLMAP_PER_GPU}")
    print(f"Total CPU cores: {config.TOTAL_CPU_CORES}")
    print(f"CPU cores per COLMAP: {config.CPU_CORES_PER_COLMAP}")
    if auto_detect:
        print(f"Auto-detection: Scanning for new materials every {config.MATERIAL_SCAN_INTERVAL_SEC}s")
    print()
    
    try:
        last_scan_time = time.time()
        iteration = 0
        
        while True:
            iteration += 1
            
            # Periodic material scanning in auto-detect mode
            if auto_detect and (time.time() - last_scan_time) >= config.MATERIAL_SCAN_INTERVAL_SEC:
                print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Scanning for new materials...")
                materials, new_count = initialize_materials(dataset_root, state, verbose=True, fix_existing=False, only=only)
                if new_count > 0:
                    save_state(state, config.STATE_FILE)
                last_scan_time = time.time()
            
            # Update status of running jobs
            update_job_status(state, config)
            
            # Try to launch shape matching for completed COLMAP jobs (priority)
            try_launch_shape_matching_jobs(state, config)

            # Disk gate: once per iteration. If tmp filesystem is below
            # MIN_FREE_DISK_GB we skip *all* COLMAP launches this cycle and
            # wait for at least one in-flight job to finish so its TMP dir
            # gets cleaned up. Shape matching is not gated (it doesn't touch
            # COLMAP_TMP_BASE).
            disk_ok, free_gb = check_disk_capacity(config)
            if not disk_ok:
                running_colmap = sum(1 for info in state["materials"].values()
                                     if info["status"] == JobStatus.COLMAP_RUNNING)
                # Throttle: only log once per ~minute to avoid spam.
                if iteration % 6 == 1:
                    print(f"[{datetime.now().strftime('%H:%M:%S')}] "
                          f"Disk gate: {free_gb:.1f} GB free at "
                          f"{config.COLMAP_TMP_BASE} < {config.MIN_FREE_DISK_GB:.0f} GB — "
                          f"waiting for {running_colmap} running COLMAP job(s) to finish")

            # Try to launch new COLMAP jobs
            for material in materials:
                info = state["materials"][material]

                if info["status"] == JobStatus.NOT_STARTED:
                    # Check if material is ready (has scan_log.json)
                    if not info.get("ready", False):
                        continue  # Skip materials that aren't ready yet

                    # Check GPU availability
                    gpu_id = get_least_loaded_gpu(state, config, gpu_monitor)
                    if gpu_id is None:
                        break  # No GPU available

                    # Check CPU capacity
                    if not check_cpu_capacity(state, config, for_shape_matching=False):
                        break  # No CPU capacity

                    # Check tmp-disk capacity (computed once above)
                    if not disk_ok:
                        break  # Wait for running COLMAP(s) to free tmp space

                    # Launch COLMAP
                    pid = launch_colmap(material, info["folder_path"], gpu_id, state, config)
                    if pid:
                        save_state(state, config.STATE_FILE)
            
            # Check if all materials are completed (or should exit)
            statuses = [info["status"] for info in state["materials"].values()]
            completed = statuses.count(JobStatus.COMPLETED)
            failed = statuses.count(JobStatus.FAILED)
            total = len(materials)
            
            # In auto-detect mode, never exit - keep waiting for new materials
            if not auto_detect and completed + failed == total:
                print("\n" + "="*60)
                print(f"All materials processed!")
                print(f"Completed: {completed}/{total}")
                print(f"Failed: {failed}/{total}")
                print("="*60 + "\n")
                break
            
            # Print status summary (every 6 iterations = ~1 minute)
            if iteration % 6 == 1:
                running_colmap = statuses.count(JobStatus.COLMAP_RUNNING)
                running_shape = statuses.count(JobStatus.SHAPE_MATCHING_RUNNING)
                not_started = statuses.count(JobStatus.NOT_STARTED)

                # Count ready vs not ready materials
                ready_count = sum(1 for info in state["materials"].values()
                                 if info["status"] == JobStatus.NOT_STARTED and info.get("ready", False))
                not_ready_count = not_started - ready_count

                warn_count = sum(1 for info in state["materials"].values()
                                 if info.get("warnings"))

                print(f"[{datetime.now().strftime('%H:%M:%S')}] Status: "
                      f"COLMAP: {running_colmap}, "
                      f"Shape: {running_shape}, "
                      f"Ready: {ready_count}, "
                      f"Not ready: {not_ready_count}, "
                      f"Completed: {completed}/{total}, "
                      f"Failed: {failed}, "
                      f"Warnings: {warn_count}")
            
            # Save state periodically
            save_state(state, config.STATE_FILE)
            
            # Wait before next check
            time.sleep(config.POLL_INTERVAL_SEC)
            
    except KeyboardInterrupt:
        print("\n\nScheduler interrupted by user. Cleaning up running processes...")
        
        # Terminate all running processes
        for material, info in state["materials"].items():
            if info["status"] in [JobStatus.COLMAP_RUNNING, JobStatus.SHAPE_MATCHING_RUNNING]:
                pid = info.get("pid")
                job_type = "COLMAP" if info["status"] == JobStatus.COLMAP_RUNNING else "shape matching"
                terminate_process(pid, material, job_type)
        
        save_state(state, config.STATE_FILE)
        print("State saved. Exiting.")
        sys.exit(0)

def manual_mode(dataset_root: str, n_folders: int, config: Config, retry_failed: bool = True, force_restart_colmap: bool = True, force_redo_ids: Optional[List[str]] = None, only: Optional[List[str]] = None):
    """
    Manual mode: Schedule first N folders and wait for all to complete.
    Balances jobs across GPUs and CPUs upfront.
    Only schedules materials that are ready (have scan_log.json).

    Args:
        dataset_root: Path to dataset root folder
        n_folders: Number of folders to process
        config: Scheduler configuration
        retry_failed: If True, reset failed jobs to retry them on startup
        force_restart_colmap: If True, always restart failed jobs from COLMAP (default True)
        force_redo_ids: If provided, force these material IDs back to NOT_STARTED
    """
    print("\n" + "="*60)
    print(f"MANUAL MODE: Scheduling first {n_folders} folders")
    if retry_failed:
        print(f"Failed jobs will be retried (restart from {'COLMAP' if force_restart_colmap else 'failed stage'})")
    print("="*60 + "\n")

    # Initialize GPU monitor
    gpu_monitor = GPUMonitor()

    state = load_state(config.STATE_FILE)
    state["mode"] = "manual"

    # Initialize materials (with failed job reset if requested)
    materials, new_count = initialize_materials(dataset_root, state, reset_failed=retry_failed, force_restart_colmap=force_restart_colmap, only=only)

    # Force-redo specific materials if requested
    if force_redo_ids:
        print(f"\nForce-redo requested for {len(force_redo_ids)} material(s):")
        n_redo = force_redo_materials(state, force_redo_ids)
        print(f"Force-reset {n_redo} material(s)\n")

    save_state(state, config.STATE_FILE)

    if not materials:
        print("No material folders found!")
        return
    
    # Select first N ready materials
    ready_materials = [m for m in materials if state["materials"][m].get("ready", False)]
    not_ready_materials = [m for m in materials if not state["materials"][m].get("ready", False)]
    
    if not_ready_materials:
        print(f"Warning: {len(not_ready_materials)} materials not ready (missing scan_log.json):")
        print(f"  {', '.join(not_ready_materials[:10])}")
        if len(not_ready_materials) > 10:
            print(f"  ... and {len(not_ready_materials) - 10} more")
        print()
    
    selected_materials = ready_materials[:n_folders]
    if len(selected_materials) < n_folders:
        print(f"Warning: Only {len(selected_materials)} ready materials available (requested {n_folders})")
    
    print(f"Selected materials: {selected_materials}\n")
    
    if not selected_materials:
        print("No ready materials to process!")
        return
    
    # Launch all N COLMAP jobs with balanced GPU assignment
    launched = 0
    for i, material in enumerate(selected_materials):
        info = state["materials"][material]
        
        if info["status"] == JobStatus.NOT_STARTED:
            # Round-robin GPU assignment
            gpu_id = config.GPU_IDS[i % len(config.GPU_IDS)]

            # Check GPU availability (memory + utilization)
            is_available, reason = gpu_monitor.is_gpu_available(
                gpu_id,
                min_memory_mb=config.MIN_GPU_MEMORY_MB,
                max_gpu_util=config.MAX_GPU_UTILIZATION,
                max_memory_util=config.MAX_MEMORY_UTILIZATION
            )
            if not is_available:
                print(f"Warning: GPU {gpu_id} unavailable ({reason}), waiting...")
                time.sleep(5)
                is_available, reason = gpu_monitor.is_gpu_available(
                    gpu_id,
                    min_memory_mb=config.MIN_GPU_MEMORY_MB,
                    max_gpu_util=config.MAX_GPU_UTILIZATION,
                    max_memory_util=config.MAX_MEMORY_UTILIZATION
                )
                if not is_available:
                    print(f"Skipping material {material} due to GPU unavailability: {reason}")
                    continue

            # Block until the tmp filesystem has MIN_FREE_DISK_GB free. In
            # manual mode we haven't started polling yet, so we sleep-and-
            # retry here instead of breaking out of the loop.
            disk_ok, free_gb = check_disk_capacity(config)
            while not disk_ok:
                print(f"[{datetime.now().strftime('%H:%M:%S')}] "
                      f"Disk gate: {free_gb:.1f} GB free at "
                      f"{config.COLMAP_TMP_BASE} < {config.MIN_FREE_DISK_GB:.0f} GB — "
                      f"waiting 30s for running COLMAP(s) to finish")
                time.sleep(30)
                disk_ok, free_gb = check_disk_capacity(config)

            # Launch COLMAP
            pid = launch_colmap(material, info["folder_path"], gpu_id, state, config)
            if pid:
                launched += 1
                save_state(state, config.STATE_FILE)
            
            # Small delay to avoid overwhelming the system
            time.sleep(2)
    
    print(f"\nLaunched {launched} COLMAP jobs\n")
    
    # Monitor until all selected materials are completed
    try:
        while True:
            # Update status
            update_job_status(state, config)
            
            # Try to launch shape matching
            try_launch_shape_matching_jobs(state, config)
            
            # Check completion status for selected materials only
            statuses = [state["materials"][m]["status"] for m in selected_materials]
            completed = statuses.count(JobStatus.COMPLETED)
            failed = statuses.count(JobStatus.FAILED)
            running_colmap = statuses.count(JobStatus.COLMAP_RUNNING)
            running_shape = statuses.count(JobStatus.SHAPE_MATCHING_RUNNING)
            
            if completed + failed == len(selected_materials):
                print("\n" + "="*60)
                print(f"All selected materials processed!")
                print(f"Completed: {completed}/{len(selected_materials)}")
                print(f"Failed: {failed}/{len(selected_materials)}")
                print("="*60 + "\n")
                break
            
            warn_count = sum(1 for m in selected_materials
                             if state["materials"][m].get("warnings"))

            print(f"[{datetime.now().strftime('%H:%M:%S')}] Status: "
                  f"COLMAP running: {running_colmap}, "
                  f"Shape matching running: {running_shape}, "
                  f"Completed: {completed}/{len(selected_materials)}, "
                  f"Failed: {failed}, "
                  f"Warnings: {warn_count}")
            
            save_state(state, config.STATE_FILE)
            time.sleep(config.POLL_INTERVAL_SEC)
            
    except KeyboardInterrupt:
        print("\n\nScheduler interrupted by user. Cleaning up running processes...")
        
        # Terminate all running processes
        for material, info in state["materials"].items():
            if info["status"] in [JobStatus.COLMAP_RUNNING, JobStatus.SHAPE_MATCHING_RUNNING]:
                pid = info.get("pid")
                job_type = "COLMAP" if info["status"] == JobStatus.COLMAP_RUNNING else "shape matching"
                terminate_process(pid, material, job_type)
        
        save_state(state, config.STATE_FILE)
        print("State saved. Exiting.")
        sys.exit(0)

# ==================== Status Display ====================

def show_status(config: Config):
    """Display current scheduler status."""
    state = load_state(config.STATE_FILE)
    
    if not state["materials"]:
        print("No materials tracked yet. Run scheduler first.")
        return
    
    print("\n" + "="*60)
    print("SCHEDULER STATUS")
    print("="*60)
    print(f"Mode: {state.get('mode', 'N/A')}")
    print(f"Last updated: {state.get('last_updated', 'N/A')}")
    print()
    
    # Group by status
    by_status = {}
    for material, info in sorted(state["materials"].items(), key=lambda x: int(x[0])):
        status = info["status"]
        if status not in by_status:
            by_status[status] = []
        by_status[status].append(material)
    
    for status in [JobStatus.COLMAP_RUNNING, JobStatus.SHAPE_MATCHING_RUNNING, 
                   JobStatus.COLMAP_DONE, JobStatus.COMPLETED, JobStatus.FAILED, 
                   JobStatus.NOT_STARTED]:
        if status in by_status:
            print(f"{status}: {len(by_status[status])} materials")
            print(f"  {', '.join(by_status[status][:20])}")
            if len(by_status[status]) > 20:
                print(f"  ... and {len(by_status[status]) - 20} more")
            print()
    
    # Show running jobs details
    print("Running Jobs:")
    for material, info in sorted(state["materials"].items(), key=lambda x: int(x[0])):
        if info["status"] in [JobStatus.COLMAP_RUNNING, JobStatus.SHAPE_MATCHING_RUNNING]:
            if info["status"] == JobStatus.COLMAP_RUNNING:
                print(f"  Material {material}: COLMAP on GPU {info.get('gpu', '?')} (PID: {info.get('pid', '?')})")
            else:
                print(f"  Material {material}: Shape matching with {info.get('workers', '?')} workers (PID: {info.get('pid', '?')})")

    # Show completed-with-warnings materials
    warn_mats = [(m, info) for m, info in
                 sorted(state["materials"].items(), key=lambda x: int(x[0]))
                 if info.get("warnings")]
    if warn_mats:
        print()
        print(f"Warnings ({len(warn_mats)} materials):")
        for material, info in warn_mats:
            codes = ",".join(w["code"] for w in info["warnings"])
            print(f"  Material {material} [{codes}]")
            for w in info["warnings"]:
                print(f"    - {w['code']}: {w['msg']}")

    print("="*60 + "\n")

# ==================== Main ====================

def main():
    parser = argparse.ArgumentParser(
        description="Job scheduler for COLMAP and shape matching",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Streaming mode (automatic scheduling)
  python job_scheduler.py --dataset /path/to/dataset --mode streaming

  # Streaming mode with auto-detection (continuously monitors for new materials)
  python job_scheduler.py --dataset /path/to/dataset --mode streaming --auto_detect

  # Manual mode (first 5 folders)
  python job_scheduler.py --dataset /path/to/dataset --mode manual --n_folders 5

  # Show current status
  python job_scheduler.py --status --dataset /path/to/dataset

  # Force redo specific materials (e.g. after recapturing)
  python job_scheduler.py --dataset /path/to/dataset --mode streaming --force_redo 42 105 210

  # Custom configuration
  python job_scheduler.py --dataset /path/to/dataset --mode streaming \\
      --max_colmap_per_gpu 8 --cpu_per_colmap 6
        """
    )
    
    parser.add_argument("--dataset", type=str, help="Path to dataset root folder")
    parser.add_argument("--mode", choices=["streaming", "manual"], help="Scheduling mode")
    parser.add_argument("--n_folders", type=int, help="Number of folders to process (manual mode only)")
    parser.add_argument("--status", action="store_true", help="Show current scheduler status")
    parser.add_argument("--auto_detect", action="store_true", 
                       help="Auto-detect new materials (streaming mode only). Continuously monitors for new material folders.")
    parser.add_argument("--no_retry_failed", action="store_true",
                       help="Don't retry failed jobs on restart (default: failed jobs are retried)")
    parser.add_argument("--retry_shape_matching_only", action="store_true",
                       help="When retrying failed jobs, only retry shape matching if COLMAP completed (default: always restart from COLMAP)")
    parser.add_argument("--force_redo", type=str, nargs="+", default=None,
                       help="Force redo COLMAP + shape matching for these material IDs (e.g. --force_redo 42 105 210). "
                            "Resets them to NOT_STARTED regardless of current status.")
    parser.add_argument("--force_redo_file", type=str, default=None,
                       help="Path to JSON file with materials to force-redo. Accepts either "
                            "a flat list [1,2,3] or an object with a 'materials' key "
                            "(e.g. redo_list.json). Merged with --force_redo.")
    parser.add_argument("--only", type=str, nargs="+", default=None,
                       help="Restrict scheduler to ONLY these material IDs. All other materials "
                            "are left completely untouched (no verify, no failed-retry, no launch). "
                            "Overrides skip.txt for the listed IDs. Example: --only 13 42")
    
    # Optional configuration overrides
    parser.add_argument("--max_colmap_per_gpu", type=int, help=f"Max COLMAP jobs per GPU (default: {Config.MAX_COLMAP_PER_GPU})")
    parser.add_argument("--cpu_per_colmap", type=int, help=f"CPU cores per COLMAP (default: {Config.CPU_CORES_PER_COLMAP})")
    parser.add_argument("--min_gpu_memory_mb", type=int, help=f"Minimum free GPU memory in MB (default: {Config.MIN_GPU_MEMORY_MB})")
    parser.add_argument("--max_gpu_util", type=int, help=f"Maximum GPU utilization %% to launch COLMAP (default: {Config.MAX_GPU_UTILIZATION})")
    parser.add_argument("--max_memory_util", type=int, help=f"Maximum memory utilization %% to launch COLMAP (default: {Config.MAX_MEMORY_UTILIZATION})")
    parser.add_argument("--state_file", type=str, help=f"State file path (default: saved in dataset folder)")
    
    args = parser.parse_args()
    
    # Create config with dataset root
    config = Config(dataset_root=args.dataset)
    
    # Apply overrides
    if args.max_colmap_per_gpu:
        config.MAX_COLMAP_PER_GPU = args.max_colmap_per_gpu
    if args.cpu_per_colmap:
        config.CPU_CORES_PER_COLMAP = args.cpu_per_colmap
    if args.min_gpu_memory_mb:
        config.MIN_GPU_MEMORY_MB = args.min_gpu_memory_mb
    if args.max_gpu_util:
        config.MAX_GPU_UTILIZATION = args.max_gpu_util
    if args.max_memory_util:
        config.MAX_MEMORY_UTILIZATION = args.max_memory_util
    if args.state_file:
        config.STATE_FILE = args.state_file
    
    # Status display mode
    if args.status:
        if not args.dataset:
            parser.error("--dataset is required for --status")
        show_status(config)
        return
    
    # Validate arguments for scheduling modes
    if not args.dataset:
        parser.error("--dataset is required when not using --status")
    if not args.mode:
        parser.error("--mode is required when not using --status")
    
    if args.mode == "manual" and not args.n_folders:
        parser.error("--n_folders is required for manual mode")
    
    if args.auto_detect and args.mode != "streaming":
        parser.error("--auto_detect can only be used with streaming mode")
    
    # Determine retry behavior (default is True, unless --no_retry_failed is specified)
    retry_failed = not args.no_retry_failed
    # Determine restart behavior (default is True = always restart from COLMAP)
    force_restart_colmap = not args.retry_shape_matching_only

    # Merge --force_redo and --force_redo_file into a single list of str IDs
    force_redo_ids = list(args.force_redo) if args.force_redo else []
    if args.force_redo_file:
        if not os.path.exists(args.force_redo_file):
            parser.error(f"--force_redo_file not found: {args.force_redo_file}")
        with open(args.force_redo_file) as f:
            payload = json.load(f)
        if isinstance(payload, list):
            file_ids = payload
        elif isinstance(payload, dict) and "materials" in payload:
            file_ids = payload["materials"]
        else:
            parser.error(f"--force_redo_file: expected a list or object with "
                         f"'materials' key, got {type(payload).__name__}")
        force_redo_ids.extend(str(m) for m in file_ids)
        print(f"Loaded {len(file_ids)} material(s) to force-redo from "
              f"{args.force_redo_file}")
    # Deduplicate while preserving order
    if force_redo_ids:
        seen = set()
        force_redo_ids = [x for x in force_redo_ids
                          if not (x in seen or seen.add(x))]
    else:
        force_redo_ids = None

    # Normalize --only IDs to strings
    only_ids = [str(m) for m in args.only] if args.only else None

    # Run scheduler
    if args.mode == "streaming":
        streaming_mode(args.dataset, config, auto_detect=args.auto_detect, retry_failed=retry_failed, force_restart_colmap=force_restart_colmap, force_redo_ids=force_redo_ids, only=only_ids)
    elif args.mode == "manual":
        manual_mode(args.dataset, args.n_folders, config, retry_failed=retry_failed, force_restart_colmap=force_restart_colmap, force_redo_ids=force_redo_ids, only=only_ids)

if __name__ == "__main__":
    main()


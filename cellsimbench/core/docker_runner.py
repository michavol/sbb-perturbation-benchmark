"""
Shared Docker container execution for CellSimBench framework.

Provides unified Docker container management for model training and inference.
"""

import docker
import logging
from pathlib import Path
from typing import Dict, List, Deque, Optional, Any
from collections import deque

log = logging.getLogger(__name__)

# Constants for log buffering
MAX_LOG_BUFFER_SIZE = 100  # Keep last 100 lines in memory for error reporting
ERROR_LOG_TAIL_SIZE = 50   # Show last 50 lines in error messages


class DockerRunner:
    """Handles Docker container execution with common patterns.
    
    Provides unified interface for running Docker containers with proper
    resource management, logging, and error handling.
    
    Attributes:
        docker_client: Docker client instance or None if Docker unavailable.
        
    Example:
        >>> runner = DockerRunner()
        >>> runner.run_container(image, command, volumes, config)
    """
    
    def __init__(self) -> None:
        """Initialize DockerRunner and check Docker availability.
        
        Attempts to connect to Docker daemon first. If not found, checks for
        Podman socket in XDG_RUNTIME_DIR and connects to it if available.
        """
        self.is_podman = False
        self.docker_client: Optional[docker.DockerClient] = None
        
        # Try standard Docker connection first
        print("DEBUG: Initializing DockerRunner with PODMAN FIX (v4)...")
        try:
            self.docker_client = docker.from_env()
            # Check if it's actually Podman masquerading
            try:
                version_info = self.docker_client.version()
                # Podman usually includes "podman" in components or version string
                for component in version_info.get('Components', []):
                    if 'podman' in component.get('Name', '').lower():
                        self.is_podman = True
                        break
                if not self.is_podman and 'podman' in version_info.get('Version', '').lower():
                    self.is_podman = True
                
                if self.is_podman:
                    log.info("Detected Podman environment")
            except Exception:
                pass
                
            log.info("Successfully connected to Docker daemon")
        except docker.errors.DockerException:
            # Try Podman socket
            import os
            xdg_runtime_dir = os.environ.get('XDG_RUNTIME_DIR')
            if xdg_runtime_dir:
                podman_socket = Path(xdg_runtime_dir) / 'podman' / 'podman.sock'
                if podman_socket.exists():
                    try:
                        # Podman provides a Docker-compatible API on this socket
                        self.docker_client = docker.DockerClient(base_url=f"unix://{podman_socket}")
                        self.is_podman = True
                        log.info(f"Docker not found, but successfully connected to Podman socket at {podman_socket}")
                    except Exception as e:
                        log.warning(f"Found Podman socket but failed to connect: {e}")
            
            if self.docker_client is None:
                log.warning("Docker not available and Podman socket not found or accessible.")
    
    def run_container(
        self,
        image: str,
        command: List[str],
        volumes: Dict[str, Dict[str, str]],
        docker_config: Dict[str, Any],
        container_name: str = "cellsimbench",
        environment: Optional[Dict[str, str]] = None,
        gpu_id: Optional[int] = None,
        entrypoint: Optional[str] = None
    ) -> None:
        """Run a Docker container with standard configuration.
        
        Handles resource limits, GPU support, volume mounting, and streaming logs.
        
        Args:
            image: Docker image name (e.g., 'cellsimbench/sclambda:latest').
            command: Command to run in container (e.g., ['train', '/config.json']).
            volumes: Volume mount configuration mapping host paths to container paths.
            docker_config: Docker settings including memory, cpus, gpu support.
            container_name: Container name for logging purposes.
            environment: Optional environment variables to pass to container.
            entrypoint: Optional entrypoint override.
            
        Raises:
            RuntimeError: If Docker is not available or container fails.
        """
        if self.docker_client is None:
            raise RuntimeError("Docker is not available")
        
        # Build container arguments
        container_args: Dict[str, Any] = {
            'image': image,
            'command': command,
            'volumes': volumes,
            'detach': True,
            'remove': False  # Don't auto-remove so we can get logs on failure
        }
        
        # Add environment variables if provided
        if environment:
            container_args['environment'] = environment
            log.info(f"Setting environment variables: {list(environment.keys())}")
            
        if entrypoint:
            container_args['entrypoint'] = entrypoint
        
        # Add memory limit if not "max"
        memory_config = docker_config.get('memory', 'max')
        if memory_config != 'max':
            container_args['mem_limit'] = memory_config
            log.info(f"Using memory limit: {memory_config}")
        else:
            log.info("Using maximum available memory (no limit)")
        
        # Add CPU limit if not "max"
        cpus_config = docker_config.get('cpus', 'max')
        if cpus_config != 'max':
            if isinstance(cpus_config, int):
                container_args['cpuset_cpus'] = f"0-{cpus_config-1}"
                log.info(f"Using CPU cores: 0-{cpus_config-1}")
            else:
                container_args['cpuset_cpus'] = str(cpus_config)
                log.info(f"Using CPU cores: {cpus_config}")
        else:
            log.info("Using all available CPU cores (no limit)")
        
        # Add GPU support if enabled
        if docker_config.get('gpu', True):
            if self.is_podman:
                # Check for CDI configuration which provides full driver injection
                import os
                cdi_spec_exists = os.path.exists('/var/run/cdi/nvidia.yaml') or \
                                  os.path.exists('/etc/cdi/nvidia.yaml') or \
                                  os.path.exists('/etc/containers/cdi/nvidia.yaml')
                
                if cdi_spec_exists:
                    # Use CDI device syntax via DeviceRequest with explicit 'cdi' driver
                    container_args['device_requests'] = [
                        docker.types.DeviceRequest(
                            device_ids=['nvidia.com/gpu=all'], 
                            driver='cdi',
                            capabilities=[['gpu']]
                        )
                    ]
                    container_args['security_opt'] = ['label=disable'] # Needed for rootless hooks/CDI
                    
                    # Fix environment variables for CDI execution
                    env = container_args.get('environment', {})
                    if env is None:
                        env = {}
                    
                    # 1. Ensure /usr/lib64 is in LD_LIBRARY_PATH because CDI mounts driver libs there,
                    #    but some images (like pytorch) don't include it by default.
                    current_ld = env.get('LD_LIBRARY_PATH', '')
                    if not current_ld:
                         # Attempt to inherit from image not easy via API without inspection
                         # so we provide a safe default including likely paths + CDI path
                         env['LD_LIBRARY_PATH'] = '/usr/lib64:/usr/local/nvidia/lib:/usr/local/nvidia/lib64:/usr/lib/x86_64-linux-gnu'
                    else:
                        env['LD_LIBRARY_PATH'] = f"/usr/lib64:{current_ld}"
                    
                    # 2. CDI forces NVIDIA_VISIBLE_DEVICES=void which confuses PyTorch.
                    #    We set CUDA_VISIBLE_DEVICES to ensure PyTorch sees them.
                    #    (Assuming all devices are visible as requested by 'nvidia.com/gpu=all')
                    #    We can't easily count them here without NVML, so we assume typical range 0-7.
                    env['CUDA_VISIBLE_DEVICES'] = '0,1,2,3,4,5,6,7' 
                    
                    container_args['environment'] = env
                    
                    log.info("Podman detected with CDI: Using DeviceRequest with driver='cdi' and device='nvidia.com/gpu=all'")
                    log.info("  - Added /usr/lib64 to LD_LIBRARY_PATH")
                    log.info("  - Set CUDA_VISIBLE_DEVICES to bypass NVIDIA_VISIBLE_DEVICES=void")
                else:
                    # Fallback: Manually map host NVIDIA devices and set env vars.
                    import glob
                    host_nvidia_devices = glob.glob('/dev/nvidia*')
                    if host_nvidia_devices:
                        # Map each device to itself
                        devices = [f"{dev}:{dev}:rwm" for dev in host_nvidia_devices]
                        container_args['devices'] = devices
                        
                        # Ensure environment variables are set
                        env = container_args.get('environment', {})
                        env['NVIDIA_VISIBLE_DEVICES'] = 'all'
                        env['NVIDIA_DRIVER_CAPABILITIES'] = 'compute,utility'
                        container_args['environment'] = env
                        
                        log.info(f"Podman detected (no CDI): Manually mapped {len(devices)} NVIDIA devices and set NVIDIA environment variables")
                        
                        # Podman (especially rootless) often needs this to allow hooks to inject files
                        # properly without SELinux blocking them.
                        container_args['security_opt'] = ['label=disable']
                    else:
                        log.warning("Podman detected and GPU requested, but no /dev/nvidia* devices found on host.")
            else:
                # Standard Docker: use DeviceRequest
                if gpu_id is not None:
                    # Assign specific GPU to this container
                    container_args['device_requests'] = [
                        docker.types.DeviceRequest(device_ids=[str(gpu_id)], driver='nvidia', capabilities=[['gpu']])
                    ]
                    log.info(f"Assigning GPU {gpu_id} to container (driver='nvidia')")
                else:
                    # Default: give access to all GPUs
                    container_args['device_requests'] = [
                        docker.types.DeviceRequest(count=-1, driver='nvidia', capabilities=[['gpu']])
                    ]
                    log.info("Assigning all available GPUs to container (driver='nvidia')")
        
        # Add shared memory configuration for PyTorch DataLoaders
        shm_size = docker_config.get('shm_size', '2g')  # Default 2GB for ML workloads
        container_args['shm_size'] = shm_size
        log.info(f"Using shared memory size: {shm_size}")
        
        log.info(f"Volume mounts:")
        for host_path, mount_config in volumes.items():
            log.info(f"  {host_path} -> {mount_config['bind']}")
        
        log.info(f"Starting {container_name} with Docker image: {image}")
        
        # Run container
        container = self.docker_client.containers.run(**container_args)
        
        # Buffer to capture recent logs for error reporting
        log_buffer: Deque[str] = deque(maxlen=MAX_LOG_BUFFER_SIZE)
        
        # Stream logs while capturing for error reporting
        for line in container.logs(stream=True, follow=True):
            decoded_line = line.decode().strip()
            
            # Log in real-time
            log.info(f"[DOCKER] {decoded_line}")
            
            # Buffer for error reporting
            log_buffer.append(decoded_line)
        
        # Wait for completion
        result = container.wait()
        
        # Clean up container
        try:
            container.remove()
        except Exception:
            pass  # Container might already be removed
        
        if result['StatusCode'] != 0:
            # Build error message using buffered logs
            error_msg = f"{container_name} failed with exit code: {result['StatusCode']}\n"
            error_msg += f"Command: {' '.join(command)}\n"
            error_msg += f"Image: {image}\n\n"
            
            # Use buffered logs for error context
            if log_buffer:
                buffered_logs = '\n'.join(log_buffer)
                error_msg += f"Container output (last {len(log_buffer)} lines):\n"
                error_msg += f"{buffered_logs}\n\n"
            
            # Add debugging hints
            error_msg += "Debugging hints:\n"
            error_msg += "- Check if all required files are mounted correctly\n"
            error_msg += "- Verify the Docker image contains all dependencies\n"
            error_msg += "- Check configuration file format and values\n"
            error_msg += "- Ensure sufficient memory/disk space is available"
            
            raise RuntimeError(error_msg)
        
        log.info(f"{container_name} completed successfully") 
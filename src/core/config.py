"""
Configuration management for GPU-optimized STT Decoder
Handles all settings for model, GPU, inference, and performance
GPU-Only Configuration - CPU support removed for performance optimization
"""

import os
import torch
from pathlib import Path
from typing import Optional, Literal
from dataclasses import dataclass, field
from enum import Enum


class ComputeType(Enum):
    """Supported compute types for GPU inference"""
    FLOAT32 = "float32"
    FLOAT16 = "float16" 
    INT8 = "int8"


@dataclass
class ModelConfig:
    """Model-specific configuration - GPU optimized"""
    name: str = "large-v3"
    cache_dir: str = "./models/whisper"
    download_root: Optional[str] = None
    local_files_only: bool = False
    device: str = "cuda"  # GPU only
    compute_type: ComputeType = ComputeType.FLOAT16  # GPU optimized default
    
    # Inference parameters
    beam_size: int = 5
    temperature: float = 0.1
    no_speech_threshold: float = 0.6
    default_language: str = "ko"
    condition_on_previous_text: bool = True
    
    # Performance settings - GPU optimized
    num_workers: int = 4
    batch_size: int = 1
    
    def __post_init__(self):
        """Validate GPU availability and optimize settings"""
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available. This is a GPU-only configuration.")
            
        # Ensure download_root defaults to cache_dir
        if self.download_root is None:
            self.download_root = self.cache_dir


@dataclass 
class GPUConfig:
    """GPU-specific configuration - RTX 4090 optimized"""
    device_id: int = 0
    memory_limit: Optional[str] = None  # e.g., "16GB"
    max_concurrent_rt: int = 15  # Real-time inference slots
    max_concurrent_batch: int = 3  # Batch inference slots
    enable_mixed_precision: bool = True
    
    # RTX 4090 specific optimizations
    enable_cuda_graphs: bool = True
    enable_tensor_cores: bool = True
    memory_pool_size: str = "8GB"
    
    def get_device_name(self) -> str:
        """Get GPU device name"""
        if torch.cuda.is_available() and self.device_id < torch.cuda.device_count():
            return torch.cuda.get_device_name(self.device_id)
        raise RuntimeError("No GPU available")
    
    def get_memory_info(self) -> dict:
        """Get GPU memory information"""
        if torch.cuda.is_available():
            return {
                "total": torch.cuda.get_device_properties(self.device_id).total_memory,
                "allocated": torch.cuda.memory_allocated(self.device_id),
                "cached": torch.cuda.memory_reserved(self.device_id)
            }
        raise RuntimeError("No GPU available")


@dataclass
class InferenceConfig:
    """Inference-specific configuration - GPU optimized"""
    # Timeouts
    rt_timeout_ms: int = 2000
    batch_timeout_ms: int = 30000
    
    # Queue settings
    queue_max_size: int = 100
    
    # Audio processing
    sample_rate: int = 16000
    max_audio_duration: float = 30.0
    min_audio_duration: float = 0.1
    audio_quality_threshold: float = 5.0  # SNR in dB
    
    # Keyword boosting
    enable_keyword_boosting: bool = True
    default_boost_factor: float = 1.5
    max_keywords_per_request: int = 100
    
    # GPU-specific optimizations
    enable_vad: bool = True
    vad_threshold: float = 0.5
    chunk_length: int = 30  # seconds


@dataclass
class ServerConfig:
    """Server configuration"""
    host: str = "0.0.0.0"
    port: int = 8080
    debug: bool = False
    log_level: str = "info"
    
    # Performance
    workers: int = 1
    max_requests: int = 1000
    keepalive_timeout: int = 5


@dataclass
class STTConfig:
    """Main configuration class combining all settings - GPU Only"""
    model: ModelConfig = field(default_factory=ModelConfig)
    gpu: GPUConfig = field(default_factory=GPUConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    server: ServerConfig = field(default_factory=ServerConfig)
    
    @classmethod
    def from_env(cls) -> 'STTConfig':
        """Create configuration from environment variables"""
        config = cls()
        
        # Model configuration from env
        config.model.name = os.getenv('MODEL_NAME', config.model.name)
        config.model.cache_dir = os.getenv('MODEL_CACHE_DIR', config.model.cache_dir)
        config.model.beam_size = int(os.getenv('DEFAULT_BEAM_SIZE', config.model.beam_size))
        config.model.temperature = float(os.getenv('DEFAULT_TEMPERATURE', config.model.temperature))
        config.model.default_language = os.getenv('DEFAULT_LANGUAGE', config.model.default_language)
        
        # GPU configuration from env
        config.gpu.device_id = int(os.getenv('CUDA_DEVICE', config.gpu.device_id))
        config.gpu.memory_limit = os.getenv('GPU_MEMORY_LIMIT', config.gpu.memory_limit)
        config.gpu.max_concurrent_rt = int(os.getenv('MAX_CONCURRENT_RT', config.gpu.max_concurrent_rt))
        config.gpu.max_concurrent_batch = int(os.getenv('MAX_CONCURRENT_BATCH', config.gpu.max_concurrent_batch))
        
        # Server configuration from env
        config.server.host = os.getenv('HOST', config.server.host)
        config.server.port = int(os.getenv('PORT', config.server.port))
        config.server.debug = os.getenv('DEBUG', '').lower() in ('true', '1', 'yes')
        config.server.log_level = os.getenv('LOG_LEVEL', config.server.log_level)
        
        return config
        
    def validate(self) -> bool:
        """Validate configuration settings - GPU only"""
        errors = []
        
        # Validate GPU availability
        if not torch.cuda.is_available():
            errors.append("CUDA is not available. GPU-only configuration requires CUDA.")
            
        # Validate model settings
        if not Path(self.model.cache_dir).exists():
            try:
                Path(self.model.cache_dir).mkdir(parents=True, exist_ok=True)
            except Exception as e:
                errors.append(f"Cannot create model cache directory: {e}")
        
        # Validate GPU settings
        if self.gpu.device_id >= torch.cuda.device_count():
            errors.append(f"GPU device {self.gpu.device_id} not available")
            
        # Validate inference settings
        if self.inference.max_audio_duration <= 0:
            errors.append("max_audio_duration must be positive")
            
        if self.inference.sample_rate <= 0:
            errors.append("sample_rate must be positive")
            
        # Validate server settings
        if not (1 <= self.server.port <= 65535):
            errors.append("port must be between 1 and 65535")
            
        if errors:
            raise ValueError("Configuration validation failed:\n" + "\n".join(errors))
            
        return True
    
    def summary(self) -> str:
        """Generate a summary of the current configuration"""
        return f"""
GPU-Optimized STT Decoder Configuration Summary:
===============================================

Model Configuration:
  - Model: {self.model.name}
  - Device: {self.model.device} (GPU Only)
  - Compute Type: {self.model.compute_type.value}
  - Cache Directory: {self.model.cache_dir}
  - Beam Size: {self.model.beam_size}
  - Language: {self.model.default_language}

GPU Configuration:
  - Device ID: {self.gpu.device_id}
  - Device Name: {self.gpu.get_device_name()}
  - Max RT Concurrent: {self.gpu.max_concurrent_rt}
  - Max Batch Concurrent: {self.gpu.max_concurrent_batch}
  - Mixed Precision: {self.gpu.enable_mixed_precision}
  - CUDA Graphs: {self.gpu.enable_cuda_graphs}
  - Tensor Cores: {self.gpu.enable_tensor_cores}

Inference Configuration:
  - Sample Rate: {self.inference.sample_rate} Hz
  - Max Audio Duration: {self.inference.max_audio_duration}s
  - RT Timeout: {self.inference.rt_timeout_ms}ms
  - Keyword Boosting: {self.inference.enable_keyword_boosting}
  - VAD Enabled: {self.inference.enable_vad}

Server Configuration:
  - Host: {self.server.host}
  - Port: {self.server.port}
  - Debug: {self.server.debug}
  - Log Level: {self.server.log_level}
"""


# Default GPU-optimized configuration instance
DEFAULT_CONFIG = STTConfig() 
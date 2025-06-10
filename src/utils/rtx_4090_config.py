"""
RTX 4090 Specific Configuration Optimizer
Fine-tuned configurations for NVIDIA GeForce RTX 4090 with CUDA 12.2
"""

import logging
import os
import warnings
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict

import torch
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class RTX4090Config:
    """RTX 4090 specific configuration parameters"""
    
    # Hardware specifications
    total_vram_gb: float = 24.0
    compute_capability: Tuple[int, int] = (8, 9)  # Ada Lovelace
    tensor_cores: int = 128
    sm_count: int = 128
    max_threads_per_sm: int = 1536
    
    # Memory optimization
    memory_pool_init_gb: float = 2.0
    memory_pool_max_gb: float = 22.0
    memory_fragmentation_threshold: float = 0.1
    vram_utilization_target: float = 0.85
    
    # Processing optimization
    optimal_batch_sizes: Dict[str, int] = None
    optimal_sequence_lengths: Dict[str, int] = None
    tensor_parallel_size: int = 1
    pipeline_parallel_size: int = 1
    
    # Whisper-specific optimization
    whisper_large_v3_optimization: Dict[str, Any] = None
    korean_language_optimization: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.optimal_batch_sizes is None:
            self.optimal_batch_sizes = {
                'single_inference': 1,
                'batch_inference': 8,
                'concurrent_streams': 4,
                'max_batch_size': 16
            }
        
        if self.optimal_sequence_lengths is None:
            self.optimal_sequence_lengths = {
                'short_audio': 3000,  # ~30 seconds at 16kHz
                'medium_audio': 9600,  # ~60 seconds at 16kHz  
                'long_audio': 19200,  # ~120 seconds at 16kHz
                'max_sequence': 48000  # ~300 seconds at 16kHz
            }
        
        if self.whisper_large_v3_optimization is None:
            self.whisper_large_v3_optimization = {
                'model_parameters': {
                    'num_workers': 6,
                    'cpu_threads': 8,
                    'device_index': 0,
                    'local_files_only': False
                },
                'inference_parameters': {
                    'beam_size': 5,
                    'best_of': 5,
                    'patience': 1.0,
                    'length_penalty': 1.0,
                    'repetition_penalty': 1.05,
                    'no_repeat_ngram_size': 3,
                    'temperature': [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
                    'compression_ratio_threshold': 2.4,
                    'log_prob_threshold': -1.0,
                    'no_speech_threshold': 0.6
                },
                'vad_parameters': {
                    'threshold': 0.35,
                    'min_speech_duration_ms': 200,
                    'max_speech_duration_s': 30,
                    'min_silence_duration_ms': 1500,
                    'speech_pad_ms': 300
                }
            }
        
        if self.korean_language_optimization is None:
            self.korean_language_optimization = {
                'vad_adjustment': {
                    'threshold_reduction': 0.1,  # Lower threshold for Korean
                    'min_speech_duration_ms': 150,
                    'speech_pad_ms': 250
                },
                'inference_adjustment': {
                    'repetition_penalty': 1.1,
                    'no_repeat_ngram_size': 4,
                    'temperature': [0.0, 0.1, 0.3, 0.5, 0.7, 0.9]
                },
                'post_processing': {
                    'normalize_korean_text': True,
                    'enhance_word_boundaries': True,
                    'filter_low_confidence': True,
                    'confidence_threshold': 0.3
                }
            }


class RTX4090Optimizer:
    """Advanced RTX 4090 optimization and configuration manager"""
    
    def __init__(self, config: Optional[RTX4090Config] = None):
        self.config = config or RTX4090Config()
        self.device = torch.device("cuda:0")
        self.gpu_properties = None
        self.optimization_state = {}
        
        self._initialize_gpu_properties()
        self._verify_hardware()
        
        logger.info(f"RTX 4090 Optimizer initialized with {self.config.total_vram_gb}GB VRAM")
    
    def _initialize_gpu_properties(self):
        """Initialize GPU properties and capabilities"""
        if torch.cuda.is_available():
            self.gpu_properties = torch.cuda.get_device_properties(0)
            logger.info(f"GPU Properties: {self.gpu_properties.name}")
            logger.info(f"  Total Memory: {self.gpu_properties.total_memory / 1024**3:.1f}GB")
            logger.info(f"  Multiprocessors: {self.gpu_properties.multi_processor_count}")
            logger.info(f"  Compute Capability: {self.gpu_properties.major}.{self.gpu_properties.minor}")
        else:
            logger.warning("CUDA not available - RTX 4090 optimizations will be limited")
    
    def _verify_hardware(self):
        """Verify that we're running on RTX 4090 or compatible hardware"""
        if not torch.cuda.is_available():
            logger.warning("⚠️ CUDA not available")
            return False
        
        gpu_name = torch.cuda.get_device_name()
        if "RTX 4090" not in gpu_name:
            logger.warning(f"⚠️ Expected RTX 4090, found: {gpu_name}")
            logger.info("Continuing with generic optimization parameters")
        
        # Verify compute capability
        if self.gpu_properties:
            cc = (self.gpu_properties.major, self.gpu_properties.minor)
            if cc < self.config.compute_capability:
                logger.warning(f"⚠️ Lower compute capability detected: {cc} < {self.config.compute_capability}")
        
        return True
    
    def optimize_memory_configuration(self) -> Dict[str, Any]:
        """Configure optimal memory settings for RTX 4090"""
        logger.info("🔧 Optimizing memory configuration for RTX 4090")
        
        optimizations = {}
        
        try:
            # Set memory fraction
            target_fraction = self.config.vram_utilization_target
            torch.cuda.set_per_process_memory_fraction(target_fraction)
            optimizations['memory_fraction'] = target_fraction
            logger.info(f"✅ Memory fraction set to {target_fraction:.2f}")
            
            # Configure memory pool
            if hasattr(torch.cuda, 'memory'):
                # Enable memory pool expansion
                os.environ['PYTORCH_CUDA_ALLOC_CONF'] = (
                    f"max_split_size_mb:{int(self.config.memory_pool_max_gb * 1024)},"
                    f"expandable_segments:True,"
                    f"backend:native"
                )
                optimizations['memory_pool_config'] = True
                logger.info("✅ CUDA memory pool configured")
            
            # Set memory management strategy
            torch.cuda.empty_cache()
            if hasattr(torch.cuda, 'set_sync_debug_mode'):
                torch.cuda.set_sync_debug_mode(0)  # Disable sync debugging for performance
            
            optimizations['cache_management'] = True
            logger.info("✅ Memory management optimized")
            
        except Exception as e:
            logger.error(f"❌ Memory configuration failed: {e}")
            optimizations['error'] = str(e)
        
        return optimizations
    
    def optimize_compute_configuration(self) -> Dict[str, Any]:
        """Configure optimal compute settings for RTX 4090"""
        logger.info("🔧 Optimizing compute configuration for RTX 4090")
        
        optimizations = {}
        
        try:
            # Enable Tensor Core optimizations
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cuda.matmul.allow_tf32 = True
            optimizations['tf32_enabled'] = True
            logger.info("✅ TF32 Tensor Core acceleration enabled")
            
            # Enable cuDNN benchmark mode for optimal kernel selection
            torch.backends.cudnn.benchmark = True
            optimizations['cudnn_benchmark'] = True
            logger.info("✅ cuDNN benchmark mode enabled")
            
            # Set deterministic mode (if needed)
            if os.getenv('DETERMINISTIC_MODE', '').lower() == 'true':
                torch.backends.cudnn.deterministic = True
                optimizations['deterministic'] = True
                logger.info("✅ Deterministic mode enabled")
            else:
                torch.backends.cudnn.deterministic = False
                optimizations['deterministic'] = False
            
            # Configure thread settings
            torch.set_num_threads(self.config.whisper_large_v3_optimization['model_parameters']['cpu_threads'])
            optimizations['cpu_threads'] = self.config.whisper_large_v3_optimization['model_parameters']['cpu_threads']
            logger.info(f"✅ CPU threads set to {optimizations['cpu_threads']}")
            
        except Exception as e:
            logger.error(f"❌ Compute configuration failed: {e}")
            optimizations['error'] = str(e)
        
        return optimizations
    
    def get_optimal_whisper_config(self, base_config: Dict[str, Any]) -> Dict[str, Any]:
        """Get RTX 4090 optimized Whisper model configuration"""
        logger.info("⚙️ Generating optimal Whisper configuration for RTX 4090")
        
        # Start with base configuration
        optimized_config = base_config.copy()
        
        # Apply RTX 4090 specific optimizations
        model_opts = self.config.whisper_large_v3_optimization['model_parameters']
        optimized_config.update({
            'device_index': model_opts['device_index'],
            'num_workers': model_opts['num_workers'],
            'cpu_threads': model_opts['cpu_threads'],
            'local_files_only': model_opts['local_files_only']
        })
        
        # Memory optimization
        optimized_config['download_root'] = os.path.expanduser("~/.cache/whisper")
        
        logger.info("✅ Whisper configuration optimized for RTX 4090")
        return optimized_config
    
    def get_optimal_inference_params(self, base_params: Dict[str, Any]) -> Dict[str, Any]:
        """Get RTX 4090 optimized inference parameters"""
        logger.info("🎯 Generating optimal inference parameters for RTX 4090")
        
        # Start with base parameters
        optimized_params = base_params.copy()
        
        # Apply RTX 4090 specific inference optimizations
        inference_opts = self.config.whisper_large_v3_optimization['inference_parameters']
        optimized_params.update(inference_opts)
        
        # Apply Korean language optimizations if Korean
        if optimized_params.get('language') == 'ko':
            korean_opts = self.config.korean_language_optimization['inference_adjustment']
            optimized_params.update(korean_opts)
            logger.info("🇰🇷 Korean language optimizations applied")
        
        # Dynamic VAD parameter optimization
        vad_opts = self.config.whisper_large_v3_optimization['vad_parameters']
        if optimized_params.get('language') == 'ko':
            korean_vad = self.config.korean_language_optimization['vad_adjustment']
            vad_opts = vad_opts.copy()
            vad_opts['threshold'] -= korean_vad['threshold_reduction']
            vad_opts['min_speech_duration_ms'] = korean_vad['min_speech_duration_ms']
            vad_opts['speech_pad_ms'] = korean_vad['speech_pad_ms']
        
        optimized_params['vad_parameters'] = vad_opts
        
        logger.info("✅ Inference parameters optimized for RTX 4090")
        return optimized_params
    
    def get_optimal_batch_size(self, audio_duration: float, concurrent_requests: int = 1) -> int:
        """Calculate optimal batch size based on audio duration and system load"""
        
        # Base batch size selection
        if audio_duration <= 10:  # Short audio
            base_batch_size = self.config.optimal_batch_sizes['single_inference']
        elif audio_duration <= 60:  # Medium audio
            base_batch_size = self.config.optimal_batch_sizes['batch_inference'] // 2
        else:  # Long audio
            base_batch_size = 1
        
        # Adjust for concurrent requests
        if concurrent_requests > 1:
            adjustment_factor = min(4 / concurrent_requests, 1.0)
            base_batch_size = max(1, int(base_batch_size * adjustment_factor))
        
        # Ensure we don't exceed memory limits
        max_batch = self.config.optimal_batch_sizes['max_batch_size']
        optimal_batch = min(base_batch_size, max_batch)
        
        logger.debug(f"Optimal batch size for {audio_duration}s audio with {concurrent_requests} concurrent requests: {optimal_batch}")
        return optimal_batch
    
    def monitor_performance_metrics(self) -> Dict[str, Any]:
        """Monitor RTX 4090 specific performance metrics"""
        metrics = {}
        
        try:
            if torch.cuda.is_available():
                # Memory metrics
                memory_allocated = torch.cuda.memory_allocated() / 1024**2  # MB
                memory_reserved = torch.cuda.memory_reserved() / 1024**2  # MB
                memory_total = torch.cuda.get_device_properties(0).total_memory / 1024**2  # MB
                
                metrics.update({
                    'memory_allocated_mb': memory_allocated,
                    'memory_reserved_mb': memory_reserved,
                    'memory_total_mb': memory_total,
                    'memory_utilization_percent': (memory_allocated / memory_total) * 100,
                    'memory_efficiency': memory_allocated / memory_reserved if memory_reserved > 0 else 1.0
                })
                
                # GPU utilization (if nvidia-ml-py is available)
                try:
                    import pynvml
                    pynvml.nvmlInit()
                    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                    
                    util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                    power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0  # Watts
                    
                    metrics.update({
                        'gpu_utilization_percent': util.gpu,
                        'memory_utilization_nvidia': util.memory,
                        'temperature_celsius': temp,
                        'power_usage_watts': power
                    })
                    
                except ImportError:
                    logger.debug("pynvml not available, skipping detailed GPU metrics")
                except Exception as e:
                    logger.debug(f"Failed to get nvidia-ml metrics: {e}")
                
                # Performance state
                metrics.update({
                    'tensor_cores_enabled': torch.backends.cuda.matmul.allow_tf32,
                    'cudnn_benchmark_enabled': torch.backends.cudnn.benchmark,
                    'compute_capability': f"{self.gpu_properties.major}.{self.gpu_properties.minor}" if self.gpu_properties else "unknown"
                })
                
        except Exception as e:
            logger.error(f"Failed to collect performance metrics: {e}")
            metrics['error'] = str(e)
        
        return metrics
    
    def apply_all_optimizations(self) -> Dict[str, Any]:
        """Apply all RTX 4090 optimizations"""
        logger.info("🚀 Applying all RTX 4090 optimizations")
        
        results = {}
        
        # Memory optimizations
        memory_results = self.optimize_memory_configuration()
        results['memory'] = memory_results
        
        # Compute optimizations  
        compute_results = self.optimize_compute_configuration()
        results['compute'] = compute_results
        
        # Store optimization state
        self.optimization_state = results
        
        # Check for any errors
        has_errors = any('error' in result for result in results.values() if isinstance(result, dict))
        
        if not has_errors:
            logger.info("✅ All RTX 4090 optimizations applied successfully")
        else:
            logger.warning("⚠️ Some RTX 4090 optimizations failed to apply")
        
        return results


# Global instance
_rtx_4090_optimizer: Optional[RTX4090Optimizer] = None


def get_rtx_4090_optimizer(config: Optional[RTX4090Config] = None) -> RTX4090Optimizer:
    """Get global RTX 4090 optimizer instance"""
    global _rtx_4090_optimizer
    if _rtx_4090_optimizer is None:
        _rtx_4090_optimizer = RTX4090Optimizer(config)
    return _rtx_4090_optimizer


def apply_rtx_4090_configuration() -> Dict[str, Any]:
    """Apply comprehensive RTX 4090 configuration"""
    optimizer = get_rtx_4090_optimizer()
    return optimizer.apply_all_optimizations()


def get_rtx_4090_whisper_config(base_config: Dict[str, Any]) -> Dict[str, Any]:
    """Get RTX 4090 optimized Whisper configuration"""
    optimizer = get_rtx_4090_optimizer()
    return optimizer.get_optimal_whisper_config(base_config)


def get_rtx_4090_inference_params(base_params: Dict[str, Any]) -> Dict[str, Any]:
    """Get RTX 4090 optimized inference parameters"""
    optimizer = get_rtx_4090_optimizer()
    return optimizer.get_optimal_inference_params(base_params)


def monitor_rtx_4090_performance() -> Dict[str, Any]:
    """Monitor RTX 4090 performance metrics"""
    optimizer = get_rtx_4090_optimizer()
    return optimizer.monitor_performance_metrics()


if __name__ == "__main__":
    # Test RTX 4090 configuration
    logging.basicConfig(level=logging.INFO)
    
    print("🔧 Testing RTX 4090 Configuration")
    print("=" * 50)
    
    # Initialize optimizer
    optimizer = get_rtx_4090_optimizer()
    
    # Apply all optimizations
    results = apply_rtx_4090_configuration()
    
    # Monitor performance
    metrics = monitor_rtx_4090_performance()
    
    print("\\n📊 Performance Metrics:")
    for key, value in metrics.items():
        if key != 'error':
            print(f"  {key}: {value}")
    
    print("\\n✅ RTX 4090 configuration test completed") 
"""
GPU Optimizer for Faster Whisper on RTX 4090
Advanced GPU optimization techniques for enhanced performance
"""

import gc
import logging
import os
import time
from contextlib import contextmanager
from typing import Dict, List, Optional, Any, Tuple

import torch
import numpy as np

logger = logging.getLogger(__name__)


class GPUOptimizer:
    """Advanced GPU optimization for Faster Whisper Large v3 on RTX 4090"""
    
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.gpu_name = torch.cuda.get_device_name() if torch.cuda.is_available() else "CPU"
        self.optimization_applied = False
        self.baseline_memory = 0
        
        # RTX 4090 specific optimizations
        self.rtx_4090_optimizations = {
            'memory_fraction': 0.95,  # Use 95% of VRAM for optimal performance
            'cudnn_benchmark': True,   # Enable cuDNN auto-tuner
            'mixed_precision': True,   # Enable mixed precision for speed
            'memory_pool': True,       # Use memory pool for efficient allocation
            'tensor_core_enabled': True,  # Leverage Tensor Cores
            'kernel_fusion': True,     # Enable kernel fusion
            'cache_management': True   # Advanced cache management
        }
        
        logger.info(f"🔧 GPU Optimizer initialized for: {self.gpu_name}")
    
    def apply_rtx_4090_optimizations(self) -> Dict[str, Any]:
        """Apply RTX 4090 specific optimizations"""
        optimizations_applied = {}
        
        if not torch.cuda.is_available():
            logger.warning("CUDA not available, skipping GPU optimizations")
            return optimizations_applied
        
        try:
            # 1. Memory Management Optimization
            if self.rtx_4090_optimizations['memory_fraction']:
                memory_fraction = self.rtx_4090_optimizations['memory_fraction']
                torch.cuda.set_per_process_memory_fraction(memory_fraction)
                optimizations_applied['memory_fraction'] = memory_fraction
                logger.info(f"✅ Memory fraction set to {memory_fraction}")
            
            # 2. cuDNN Benchmark Optimization
            if self.rtx_4090_optimizations['cudnn_benchmark']:
                torch.backends.cudnn.benchmark = True
                torch.backends.cudnn.deterministic = False
                optimizations_applied['cudnn_benchmark'] = True
                logger.info("✅ cuDNN benchmark enabled")
            
            # 3. Mixed Precision Setup
            if self.rtx_4090_optimizations['mixed_precision']:
                # Enable automatic mixed precision
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
                optimizations_applied['mixed_precision'] = True
                logger.info("✅ Mixed precision (TF32) enabled")
            
            # 4. Memory Pool Configuration
            if self.rtx_4090_optimizations['memory_pool']:
                # Configure memory pool for efficient allocation
                os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'
                optimizations_applied['memory_pool'] = True
                logger.info("✅ CUDA memory pool optimized")
            
            # 5. Tensor Core Optimization
            if self.rtx_4090_optimizations['tensor_core_enabled']:
                # Enable Tensor Core usage
                torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True
                optimizations_applied['tensor_core'] = True
                logger.info("✅ Tensor Core optimizations enabled")
            
            # 6. Cache Management
            if self.rtx_4090_optimizations['cache_management']:
                # Clear any existing cache
                torch.cuda.empty_cache()
                # Synchronize for clean state
                torch.cuda.synchronize()
                optimizations_applied['cache_management'] = True
                logger.info("✅ Cache management optimized")
            
            self.optimization_applied = True
            logger.info("🚀 All RTX 4090 optimizations applied successfully")
            
        except Exception as e:
            logger.error(f"❌ Error applying optimizations: {e}")
            optimizations_applied['error'] = str(e)
        
        return optimizations_applied
    
    def get_optimal_batch_size(self, audio_length_seconds: float = 30.0) -> int:
        """Calculate optimal batch size for RTX 4090"""
        if not torch.cuda.is_available():
            return 1
        
        # RTX 4090 has 24GB VRAM - calculate based on model size and audio length
        available_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        
        # Faster Whisper Large v3 uses approximately 3-4GB base + per-audio overhead
        base_model_memory_gb = 4.0
        per_audio_memory_gb = audio_length_seconds * 0.02  # Rough estimate
        
        # Calculate safe batch size (leave 20% headroom)
        available_for_batch = (available_memory_gb - base_model_memory_gb) * 0.8
        optimal_batch_size = max(1, int(available_for_batch / per_audio_memory_gb))
        
        # RTX 4090 sweet spot is typically 4-8 concurrent processes
        optimal_batch_size = min(optimal_batch_size, 8)
        
        logger.info(f"📊 Optimal batch size for {audio_length_seconds}s audio: {optimal_batch_size}")
        return optimal_batch_size
    
    def configure_whisper_model(self, model_params: Dict[str, Any]) -> Dict[str, Any]:
        """Configure Whisper model parameters for RTX 4090 optimization"""
        optimized_params = model_params.copy()
        
        # RTX 4090 optimized parameters
        rtx_4090_params = {
            # Compute optimization
            'compute_type': 'float16',  # Use FP16 for Tensor Cores
            'device': 'cuda',
            'device_index': 0,
            
            # Memory optimization
            'download_root': None,
            'local_files_only': False,
            
            # Performance optimization
            'num_workers': 8,  # RTX 4090 can handle 8 parallel workers efficiently
            'cpu_threads': min(os.cpu_count(), 16),  # Optimal CPU thread count
        }
        
        # Update with RTX 4090 optimizations
        optimized_params.update(rtx_4090_params)
        
        logger.info("⚙️ Whisper model parameters optimized for RTX 4090")
        return optimized_params
    
    def optimize_inference_parameters(self, inference_params: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize inference parameters for RTX 4090"""
        optimized_params = inference_params.copy()
        
        # RTX 4090 optimized inference parameters
        rtx_4090_inference = {
            # Speed optimizations
            'beam_size': 1,  # Faster with minimal quality loss
            'best_of': 1,    # Single candidate for speed
            'temperature': 0.0,  # Deterministic output
            
            # Memory optimizations
            'compression_ratio_threshold': 2.4,
            'log_prob_threshold': -1.0,
            'no_speech_threshold': 0.6,
            
            # Quality optimizations
            'condition_on_previous_text': True,
            'prepend_punctuations': "\"'([{-",
            'append_punctuations': "\"'.,:!?)]}-",
            
            # Performance optimizations
            'vad_filter': True,
            'vad_parameters': {
                'threshold': 0.3,  # Balanced threshold for RTX 4090
                'min_speech_duration_ms': 250,
                'max_speech_duration_s': 30,
                'min_silence_duration_ms': 100,
                'speech_pad_ms': 30
            }
        }
        
        # Update with optimizations
        optimized_params.update(rtx_4090_inference)
        
        logger.info("🎯 Inference parameters optimized for RTX 4090")
        return optimized_params
    
    @contextmanager
    def optimized_inference_context(self):
        """Context manager for optimized inference"""
        if not torch.cuda.is_available():
            yield
            return
        
        # Save initial state
        initial_cache_size = torch.cuda.memory_cached()
        
        try:
            # Pre-inference optimizations
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            
            # Set optimal memory settings
            with torch.cuda.amp.autocast(enabled=True):
                yield
            
        finally:
            # Post-inference cleanup
            torch.cuda.synchronize()
            
            # Smart cache management
            current_cache = torch.cuda.memory_cached()
            if current_cache > initial_cache_size * 1.5:
                torch.cuda.empty_cache()
                logger.debug("🧹 Cache cleared due to excessive growth")
    
    def monitor_gpu_performance(self) -> Dict[str, float]:
        """Monitor real-time GPU performance metrics"""
        if not torch.cuda.is_available():
            return {}
        
        metrics = {}
        
        try:
            # Memory metrics
            memory_allocated = torch.cuda.memory_allocated() / 1024**2  # MB
            memory_reserved = torch.cuda.memory_reserved() / 1024**2   # MB
            memory_cached = torch.cuda.memory_cached() / 1024**2       # MB
            
            total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**2
            memory_utilization = (memory_allocated / total_memory) * 100
            
            metrics.update({
                'memory_allocated_mb': memory_allocated,
                'memory_reserved_mb': memory_reserved,
                'memory_cached_mb': memory_cached,
                'total_memory_mb': total_memory,
                'memory_utilization_percent': memory_utilization
            })
            
            # Performance metrics
            current_device = torch.cuda.current_device()
            gpu_name = torch.cuda.get_device_name(current_device)
            
            metrics.update({
                'gpu_device_id': current_device,
                'gpu_name': gpu_name,
                'cuda_version': torch.version.cuda,
                'optimization_applied': self.optimization_applied
            })
            
        except Exception as e:
            logger.warning(f"GPU monitoring error: {e}")
            metrics['error'] = str(e)
        
        return metrics
    
    def get_optimization_recommendations(self, current_performance: Dict[str, float]) -> List[str]:
        """Get specific optimization recommendations based on current performance"""
        recommendations = []
        
        if not torch.cuda.is_available():
            recommendations.append("CUDA not available - consider GPU setup")
            return recommendations
        
        # Memory utilization analysis
        memory_util = current_performance.get('memory_utilization_percent', 0)
        if memory_util < 30:
            recommendations.append("Low memory utilization - consider increasing batch size")
        elif memory_util > 90:
            recommendations.append("High memory utilization - consider reducing batch size")
        
        # RTF analysis
        rtf = current_performance.get('rtf', float('inf'))
        if rtf > 0.1:
            recommendations.append("RTF > 0.1x - enable mixed precision and optimize parameters")
        elif rtf > 0.05:
            recommendations.append("RTF > 0.05x - fine-tune VAD parameters and beam size")
        
        # Cache analysis
        cached_mb = current_performance.get('memory_cached_mb', 0)
        allocated_mb = current_performance.get('memory_allocated_mb', 0)
        if cached_mb > allocated_mb * 2:
            recommendations.append("Excessive cache usage - implement periodic cache clearing")
        
        # GPU optimization status
        if not self.optimization_applied:
            recommendations.append("RTX 4090 optimizations not applied - run apply_rtx_4090_optimizations()")
        
        return recommendations
    
    def benchmark_gpu_operations(self) -> Dict[str, float]:
        """Benchmark GPU operations for optimization validation"""
        if not torch.cuda.is_available():
            return {'error': 'CUDA not available'}
        
        benchmarks = {}
        
        try:
            # Memory transfer benchmark
            start_time = time.time()
            test_tensor = torch.randn(1000, 1000).cuda()
            torch.cuda.synchronize()
            memory_transfer_time = time.time() - start_time
            benchmarks['memory_transfer_ms'] = memory_transfer_time * 1000
            
            # Matrix multiplication benchmark (Tensor Core test)
            start_time = time.time()
            with torch.cuda.amp.autocast():
                result = torch.matmul(test_tensor, test_tensor.T)
            torch.cuda.synchronize()
            matmul_time = time.time() - start_time
            benchmarks['tensor_core_matmul_ms'] = matmul_time * 1000
            
            # Memory allocation benchmark
            start_time = time.time()
            temp_tensors = [torch.randn(100, 100).cuda() for _ in range(100)]
            torch.cuda.synchronize()
            allocation_time = time.time() - start_time
            benchmarks['memory_allocation_ms'] = allocation_time * 1000
            
            # Cleanup
            del test_tensor, result, temp_tensors
            torch.cuda.empty_cache()
            
            logger.info("📊 GPU benchmark completed successfully")
            
        except Exception as e:
            logger.error(f"Benchmark failed: {e}")
            benchmarks['error'] = str(e)
        
        return benchmarks


# Global optimizer instance
_gpu_optimizer = None


def get_gpu_optimizer() -> GPUOptimizer:
    """Get global GPU optimizer instance"""
    global _gpu_optimizer
    if _gpu_optimizer is None:
        _gpu_optimizer = GPUOptimizer()
    return _gpu_optimizer


def apply_rtx_4090_optimizations() -> Dict[str, Any]:
    """Apply RTX 4090 optimizations globally"""
    optimizer = get_gpu_optimizer()
    return optimizer.apply_rtx_4090_optimizations()


def get_optimal_batch_size(audio_length_seconds: float = 30.0) -> int:
    """Get optimal batch size for given audio length"""
    optimizer = get_gpu_optimizer()
    return optimizer.get_optimal_batch_size(audio_length_seconds)


def configure_whisper_for_rtx_4090(model_params: Dict[str, Any]) -> Dict[str, Any]:
    """Configure Whisper model for RTX 4090"""
    optimizer = get_gpu_optimizer()
    return optimizer.configure_whisper_model(model_params)


def optimize_inference_for_rtx_4090(inference_params: Dict[str, Any]) -> Dict[str, Any]:
    """Optimize inference parameters for RTX 4090"""
    optimizer = get_gpu_optimizer()
    return optimizer.optimize_inference_parameters(inference_params) 
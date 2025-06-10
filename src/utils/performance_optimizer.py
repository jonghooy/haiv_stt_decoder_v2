#!/usr/bin/env python3
"""
Performance Optimization Module for STT Service
Advanced performance tuning, monitoring, and optimization
"""

import gc
import logging
import time
import psutil
import threading
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field
from collections import deque
import asyncio
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import torch

logger = logging.getLogger(__name__)

@dataclass
class PerformanceMetrics:
    """Performance metrics tracking"""
    timestamp: float
    rtf: float
    latency_ms: float
    audio_duration: float
    memory_usage_mb: float
    gpu_memory_mb: float
    cpu_percent: float
    thread_count: int
    meets_targets: bool
    error_count: int = 0

@dataclass
class OptimizationConfig:
    """Configuration for performance optimization"""
    # Memory optimization
    enable_gc_optimization: bool = True
    gpu_memory_fraction: float = 0.8
    enable_memory_monitoring: bool = True
    
    # Threading optimization
    max_worker_threads: int = 6
    thread_timeout: float = 30.0
    
    # Model optimization
    enable_model_caching: bool = True
    enable_dynamic_batching: bool = False  # For future batching support
    
    # Monitoring
    metrics_window_size: int = 100
    enable_detailed_logging: bool = True
    alert_threshold_rtf: float = 0.1  # Alert if RTF > 10%
    alert_threshold_latency_ms: float = 1500  # Alert if latency > 1.5s
    
    # Error handling
    max_retry_attempts: int = 3
    retry_delay_ms: float = 100
    enable_graceful_degradation: bool = True


class PerformanceMonitor:
    """Real-time performance monitoring and alerting"""
    
    def __init__(self, config: OptimizationConfig):
        self.config = config
        self.metrics_history: deque = deque(maxlen=config.metrics_window_size)
        self.alerts: List[str] = []
        self.lock = threading.Lock()
        
        # System info
        self.cpu_count = psutil.cpu_count()
        self.total_memory = psutil.virtual_memory().total / (1024**3)  # GB
        
        logger.info(f"Performance Monitor initialized - CPU cores: {self.cpu_count}, RAM: {self.total_memory:.1f}GB")
    
    def record_metrics(self, metrics: PerformanceMetrics):
        """Record performance metrics"""
        with self.lock:
            self.metrics_history.append(metrics)
            
            # Check for alerts
            self._check_alerts(metrics)
    
    def _check_alerts(self, metrics: PerformanceMetrics):
        """Check for performance alerts"""
        alerts = []
        
        if metrics.rtf > self.config.alert_threshold_rtf:
            alerts.append(f"High RTF detected: {metrics.rtf:.3f}x (threshold: {self.config.alert_threshold_rtf:.3f}x)")
        
        if metrics.latency_ms > self.config.alert_threshold_latency_ms:
            alerts.append(f"High latency detected: {metrics.latency_ms:.0f}ms (threshold: {self.config.alert_threshold_latency_ms:.0f}ms)")
        
        if metrics.memory_usage_mb > self.total_memory * 800:  # 80% of total RAM
            alerts.append(f"High memory usage: {metrics.memory_usage_mb:.0f}MB ({metrics.memory_usage_mb/self.total_memory/10:.1f}% of total)")
        
        if metrics.cpu_percent > 90:
            alerts.append(f"High CPU usage: {metrics.cpu_percent:.1f}%")
        
        for alert in alerts:
            logger.warning(f"🚨 PERFORMANCE ALERT: {alert}")
            self.alerts.append(f"{time.time()}: {alert}")
    
    def get_current_stats(self) -> Dict[str, Any]:
        """Get current performance statistics"""
        with self.lock:
            if not self.metrics_history:
                return {"error": "No metrics available"}
            
            recent_metrics = list(self.metrics_history)[-10:]  # Last 10 measurements
            
            avg_rtf = sum(m.rtf for m in recent_metrics) / len(recent_metrics)
            avg_latency = sum(m.latency_ms for m in recent_metrics) / len(recent_metrics)
            avg_memory = sum(m.memory_usage_mb for m in recent_metrics) / len(recent_metrics)
            targets_met = sum(1 for m in recent_metrics if m.meets_targets)
            
            return {
                "total_measurements": len(self.metrics_history),
                "recent_avg_rtf": avg_rtf,
                "recent_avg_latency_ms": avg_latency,
                "recent_avg_memory_mb": avg_memory,
                "targets_met_ratio": targets_met / len(recent_metrics),
                "active_alerts": len([a for a in self.alerts if time.time() - float(a.split(':')[0]) < 300]),  # Last 5 minutes
                "system_info": {
                    "cpu_cores": self.cpu_count,
                    "total_memory_gb": self.total_memory
                }
            }
    
    def get_detailed_report(self) -> Dict[str, Any]:
        """Get detailed performance report"""
        with self.lock:
            if not self.metrics_history:
                return {"error": "No metrics available"}
            
            metrics = list(self.metrics_history)
            
            rtfs = [m.rtf for m in metrics]
            latencies = [m.latency_ms for m in metrics]
            memory_usage = [m.memory_usage_mb for m in metrics]
            
            return {
                "summary": {
                    "total_requests": len(metrics),
                    "avg_rtf": sum(rtfs) / len(rtfs),
                    "min_rtf": min(rtfs),
                    "max_rtf": max(rtfs),
                    "avg_latency_ms": sum(latencies) / len(latencies),
                    "min_latency_ms": min(latencies),
                    "max_latency_ms": max(latencies),
                    "targets_met": sum(1 for m in metrics if m.meets_targets),
                    "target_success_rate": sum(1 for m in metrics if m.meets_targets) / len(metrics) * 100
                },
                "resource_usage": {
                    "avg_memory_mb": sum(memory_usage) / len(memory_usage),
                    "max_memory_mb": max(memory_usage),
                    "avg_cpu_percent": sum(m.cpu_percent for m in metrics) / len(metrics)
                },
                "recent_alerts": self.alerts[-10:],  # Last 10 alerts
                "timestamp": time.time()
            }


class PerformanceOptimizer:
    """Advanced performance optimization for STT service"""
    
    def __init__(self, config: OptimizationConfig):
        self.config = config
        self.monitor = PerformanceMonitor(config)
        self.optimization_cache = {}
        
        # Initialize optimizations
        if config.enable_gc_optimization:
            self._setup_gc_optimization()
        
        if torch.cuda.is_available() and config.gpu_memory_fraction < 1.0:
            self._setup_gpu_memory_optimization()
    
    def _setup_gc_optimization(self):
        """Setup garbage collection optimization"""
        # Set more aggressive garbage collection
        gc.set_threshold(400, 5, 5)  # More frequent collection
        logger.info("🚀 Garbage collection optimization enabled")
    
    def _setup_gpu_memory_optimization(self):
        """Setup GPU memory optimization"""
        if torch.cuda.is_available():
            # Set memory fraction
            torch.cuda.set_per_process_memory_fraction(self.config.gpu_memory_fraction)
            logger.info(f"🚀 GPU memory fraction set to {self.config.gpu_memory_fraction}")
    
    def optimize_thread_pool(self, executor: ThreadPoolExecutor) -> ThreadPoolExecutor:
        """Optimize thread pool configuration"""
        # Check current thread count
        current_threads = executor._max_workers
        optimal_threads = min(self.config.max_worker_threads, max(2, psutil.cpu_count() - 1))
        
        if current_threads != optimal_threads:
            logger.info(f"🚀 Optimizing thread pool: {current_threads} → {optimal_threads} threads")
            # Note: ThreadPoolExecutor max_workers can't be changed after creation
            # This is informational for future optimizations
        
        return executor
    
    def pre_inference_optimization(self):
        """Perform pre-inference optimizations"""
        optimizations_applied = []
        
        # Force garbage collection before inference
        if self.config.enable_gc_optimization:
            gc.collect()
            optimizations_applied.append("GC")
        
        # Clear CUDA cache if available
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            optimizations_applied.append("CUDA_CACHE")
        
        if optimizations_applied:
            logger.debug(f"Pre-inference optimizations: {', '.join(optimizations_applied)}")
    
    def post_inference_cleanup(self):
        """Perform post-inference cleanup"""
        if torch.cuda.is_available():
            # Clear CUDA cache periodically
            torch.cuda.empty_cache()
    
    def get_current_system_metrics(self) -> Dict[str, float]:
        """Get current system resource metrics"""
        # CPU usage
        cpu_percent = psutil.cpu_percent(interval=0.1)
        
        # Memory usage
        memory = psutil.virtual_memory()
        memory_mb = (memory.total - memory.available) / (1024**2)
        
        # GPU memory if available
        gpu_memory_mb = 0
        if torch.cuda.is_available():
            gpu_memory_mb = torch.cuda.memory_allocated() / (1024**2)
        
        # Thread count
        thread_count = threading.active_count()
        
        return {
            "cpu_percent": cpu_percent,
            "memory_usage_mb": memory_mb,
            "gpu_memory_mb": gpu_memory_mb,
            "thread_count": thread_count
        }
    
    def create_performance_metrics(self, rtf: float, latency_ms: float, 
                                 audio_duration: float, meets_targets: bool,
                                 error_count: int = 0) -> PerformanceMetrics:
        """Create performance metrics object"""
        system_metrics = self.get_current_system_metrics()
        
        metrics = PerformanceMetrics(
            timestamp=time.time(),
            rtf=rtf,
            latency_ms=latency_ms,
            audio_duration=audio_duration,
            memory_usage_mb=system_metrics["memory_usage_mb"],
            gpu_memory_mb=system_metrics["gpu_memory_mb"],
            cpu_percent=system_metrics["cpu_percent"],
            thread_count=system_metrics["thread_count"],
            meets_targets=meets_targets,
            error_count=error_count
        )
        
        # Record metrics
        self.monitor.record_metrics(metrics)
        
        return metrics
    
    def adaptive_parameter_tuning(self, recent_performance: List[PerformanceMetrics]) -> Dict[str, Any]:
        """Suggest adaptive parameter tuning based on recent performance"""
        if len(recent_performance) < 5:
            return {}
        
        suggestions = {}
        
        # Analyze RTF performance
        avg_rtf = sum(m.rtf for m in recent_performance) / len(recent_performance)
        
        if avg_rtf > 0.08:  # If consistently above 8% RTF
            suggestions["beam_size"] = "reduce"  # Reduce beam size for speed
            suggestions["temperature"] = "simplify"  # Use simpler temperature settings
        elif avg_rtf < 0.03:  # If consistently very fast
            suggestions["beam_size"] = "increase"  # Can afford higher quality
            suggestions["temperature"] = "expand"  # Use more temperature options
        
        # Analyze memory usage
        avg_memory = sum(m.memory_usage_mb for m in recent_performance) / len(recent_performance)
        if avg_memory > self.monitor.total_memory * 700:  # > 70% memory usage
            suggestions["batch_size"] = "reduce"
            suggestions["gc_frequency"] = "increase"
        
        # Analyze error patterns
        total_errors = sum(m.error_count for m in recent_performance)
        if total_errors > 0:
            suggestions["retry_strategy"] = "enhance"
            suggestions["timeout"] = "increase"
        
        return suggestions
    
    def emergency_optimization(self):
        """Emergency optimization when performance degrades"""
        logger.warning("🆘 Applying emergency optimization")
        
        # Aggressive cleanup
        gc.collect()
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        # Clear caches
        self.optimization_cache.clear()
        
        logger.info("✅ Emergency optimization completed")


class ErrorHandler:
    """Enhanced error handling with recovery mechanisms"""
    
    def __init__(self, config: OptimizationConfig):
        self.config = config
        self.error_counts = {}
        self.recovery_strategies = {}
        
    def handle_error(self, error: Exception, context: str) -> Tuple[bool, Optional[str]]:
        """
        Handle error with recovery strategy
        
        Returns:
            (should_retry, recovery_message)
        """
        error_type = type(error).__name__
        error_key = f"{context}:{error_type}"
        
        # Track error count
        self.error_counts[error_key] = self.error_counts.get(error_key, 0) + 1
        
        logger.error(f"Error in {context}: {error_type} - {str(error)}")
        
        # Determine recovery strategy
        should_retry = False
        recovery_message = None
        
        if self.error_counts[error_key] <= self.config.max_retry_attempts:
            # Common recovery strategies
            if "CUDA" in str(error) or "memory" in str(error).lower():
                # GPU memory issues
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()
                should_retry = True
                recovery_message = "GPU memory cleared, retrying"
                
            elif "timeout" in str(error).lower():
                # Timeout issues
                should_retry = True
                recovery_message = f"Timeout recovery, attempt {self.error_counts[error_key]}"
                
            elif "connection" in str(error).lower():
                # Connection issues
                should_retry = True
                recovery_message = "Connection retry"
                
            else:
                # Generic retry for other errors
                if self.error_counts[error_key] <= 2:
                    should_retry = True
                    recovery_message = f"Generic retry {self.error_counts[error_key]}"
        
        # Apply graceful degradation if enabled
        if not should_retry and self.config.enable_graceful_degradation:
            recovery_message = self._apply_graceful_degradation(error_type, context)
        
        return should_retry, recovery_message
    
    def _apply_graceful_degradation(self, error_type: str, context: str) -> str:
        """Apply graceful degradation strategies"""
        strategies = []
        
        if "transcribe" in context.lower():
            strategies.append("Reduced quality mode available")
        
        if error_type in ["OutOfMemoryError", "CUDAOutOfMemoryError"]:
            strategies.append("Consider smaller audio chunks")
        
        return "; ".join(strategies) if strategies else "No degradation strategy available"
    
    def get_error_summary(self) -> Dict[str, Any]:
        """Get error handling summary"""
        total_errors = sum(self.error_counts.values())
        
        return {
            "total_errors": total_errors,
            "error_types": dict(self.error_counts),
            "most_common": max(self.error_counts.items(), key=lambda x: x[1]) if self.error_counts else None,
            "config": {
                "max_retry_attempts": self.config.max_retry_attempts,
                "retry_delay_ms": self.config.retry_delay_ms,
                "graceful_degradation": self.config.enable_graceful_degradation
            }
        }


# Global optimizer instance
_optimizer: Optional[PerformanceOptimizer] = None
_error_handler: Optional[ErrorHandler] = None

def get_performance_optimizer(config: Optional[OptimizationConfig] = None) -> PerformanceOptimizer:
    """Get global performance optimizer instance"""
    global _optimizer
    if _optimizer is None:
        _optimizer = PerformanceOptimizer(config or OptimizationConfig())
    return _optimizer

def get_error_handler(config: Optional[OptimizationConfig] = None) -> ErrorHandler:
    """Get global error handler instance"""
    global _error_handler
    if _error_handler is None:
        _error_handler = ErrorHandler(config or OptimizationConfig())
    return _error_handler


if __name__ == "__main__":
    # Test performance optimization
    config = OptimizationConfig()
    optimizer = PerformanceOptimizer(config)
    
    print("🚀 Performance Optimizer Test")
    print("=" * 40)
    
    # Test system metrics
    system_metrics = optimizer.get_current_system_metrics()
    print(f"System Metrics:")
    for key, value in system_metrics.items():
        print(f"  {key}: {value}")
    
    # Test performance metrics creation
    test_metrics = optimizer.create_performance_metrics(
        rtf=0.045,
        latency_ms=350,
        audio_duration=3.0,
        meets_targets=True
    )
    
    print(f"\nTest Metrics Created:")
    print(f"  RTF: {test_metrics.rtf:.3f}x")
    print(f"  Latency: {test_metrics.latency_ms:.0f}ms")
    print(f"  Memory: {test_metrics.memory_usage_mb:.0f}MB")
    print(f"  Meets targets: {test_metrics.meets_targets}")
    
    # Test error handler
    error_handler = ErrorHandler(config)
    
    test_error = RuntimeError("Test error for recovery")
    should_retry, message = error_handler.handle_error(test_error, "test_context")
    
    print(f"\nError Handler Test:")
    print(f"  Should retry: {should_retry}")
    print(f"  Recovery message: {message}")
    
    print("\n✅ Performance optimization test completed") 
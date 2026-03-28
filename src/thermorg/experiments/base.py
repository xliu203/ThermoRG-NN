# SPDX-License-Identifier: Apache-2.0

import time
import logging
from pathlib import Path
from abc import ABC, abstractmethod

from ..utils.logging import setup_logger


class BaseExperiment(ABC):
    """实验基类，支持每5分钟 heartbeat 日志"""
    
    def __init__(self, name: str, results_dir: str = "results"):
        self.name = name
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True)
        self.logger = setup_logger(name)
        self.start_time = time.time()
        self.last_heartbeat = time.time()
        self.heartbeat_interval = 300  # 5 minutes = 300 seconds
    
    def heartbeat(self, message: str = ""):
        """每5分钟输出 heartbeat"""
        now = time.time()
        elapsed = now - self.start_time
        if now - self.last_heartbeat >= self.heartbeat_interval:
            self.logger.info(f"[HEARTBEAT {elapsed/60:.1f}min] {message}")
            self.last_heartbeat = now
    
    def save_results(self, results: dict):
        """保存结果到文件"""
        import json
        result_file = self.results_dir / f"{self.name}_results.json"
        with open(result_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        self.logger.info(f"Results saved to {result_file}")
        return result_file
    
    @abstractmethod
    def run(self):
        pass

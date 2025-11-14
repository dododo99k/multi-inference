import numpy as np
import torch



class Task():
    def __init__(self, model_id=-1, tk_id=-1, arrive_time = -1, start_time = -1, finish_time = -1,):
        # self.model_name = model_name
        self.model_id = model_id
        self.tk_id = tk_id
        self.input_data = torch.randn(1, 3, 224, 224)
        self.result_data = None
        self.arrive_time = arrive_time
        self.start_time = start_time
        self.finish_time = finish_time
        self.infer_time = -1
        self.queue_time = -1
        self.batchsize = 1

    def finalize(self, if_print=True):
        self.infer_time = self.finish_time - self.start_time
        self.queue_time = self.start_time - self.arrive_time
        if if_print: print('infer time: ', self.infer_time, 'queue time: ', self.queue_time)
        
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, Optional, List


@dataclass
class InferenceRecord():
    task_id: str
    model_name: str
    batch_size: int
    ts_ms: int # = field(default_factory=lambda: time.time_ns() // 1_000_000)
    deleted: bool = False   # 惰性删除标记


class FastRollingWindow():
    def __init__(self, window_ms: int = 120_000):
        """
        :param window_ms: 窗口大小，毫秒
        """
        self.window_ms = window_ms
        self._dq: Deque[InferenceRecord] = deque()
        self._index: Dict[str, InferenceRecord] = {}

    def _now_ms(self) -> int:
        # 用 ns 换算成 ms，精度够
        return time.time_ns() // 1_000_000

    def _evict_expired(self):
        """把时间窗外的、或者已经标记删除的，从左边弹掉"""
        now = self._now_ms()
        cutoff = now - self.window_ms
        dq = self._dq
        while dq:
            head = dq[0]
            # 过期 或 已经被标记删除了
            if head.ts_ms < cutoff or head.deleted:
                dq.popleft()
                # 如果是过期的，而且它还在索引里，要把索引也删掉
                if head.task_id in self._index and self._index[head.task_id] is head:
                    del self._index[head.task_id]
            else:
                break

    def add_task(self, task_id: str, model_name: str, batch_size: int):
        self._evict_expired()
        rec = InferenceRecord(
            task_id=task_id,
            model_name=model_name,
            batch_size=batch_size
        )
        self._dq.append(rec)
        self._index[task_id] = rec

    def finish_task(self, task_id: str):
        """
        毫秒级删除: O(1) 找到记录并标记删除，不在 deque 中间挖洞
        """
        self._evict_expired()
        rec = self._index.pop(task_id, None)
        if rec is not None:
            rec.deleted = True  # 真正清理会在 _evict_expired 里做

    def get_current_tasks(self) -> List[InferenceRecord]:
        """
        查看当前窗口内的记录
        """
        self._evict_expired()
        # 只返回未删除的
        return [r for r in self._dq if not r.deleted]

    def total_batch(self) -> int:
        self._evict_expired()
        return sum(r.batch_size for r in self._dq if not r.deleted)
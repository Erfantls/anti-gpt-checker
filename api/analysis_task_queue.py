import asyncio
import uuid
import os
from typing import Coroutine, Dict

class AnalysisTaskQueue:
    def __init__(self, max_concurrent_tasks: int = 1):
        self.queue = asyncio.Queue()
        self._worker_tasks = []
        self._task_lookup: Dict[str, int] = {}
        self.max_concurrent_tasks = max_concurrent_tasks

    def start_worker(self):
        if not self._worker_tasks:  # Only start if not already started
            for _ in range(self.max_concurrent_tasks):
                task = asyncio.create_task(self._worker())
                self._worker_tasks.append(task)

    async def enqueue(self, coro: Coroutine) -> str:
        task_id = str(uuid.uuid4())
        await self.queue.put((task_id, coro))
        return task_id

    def get_position(self, task_id: str) -> int:
        queue_items = list(self.queue._queue)
        for idx, (tid, _) in enumerate(queue_items):
            if tid == task_id:
                return idx + 1
        return 0

    async def _worker(self):
        while True:
            task_id, task_coro = await self.queue.get()
            try:
                await task_coro
            except Exception as e:
                print(f"[Analysis Queue] Task {task_id} failed: {e}")
            finally:
                self.queue.task_done()

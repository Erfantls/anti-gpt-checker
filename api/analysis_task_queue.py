import asyncio
import uuid
from typing import Callable, Coroutine, Tuple, Dict

class AnalysisTaskQueue:
    def __init__(self):
        self.queue = asyncio.Queue()
        self._worker_task = None
        self._task_lookup: Dict[str, int] = {}  # task_id -> queue index (updated on enqueue)

    def start_worker(self):
        if self._worker_task is None:
            self._worker_task = asyncio.create_task(self._worker())

    async def enqueue(self, coro: Coroutine) -> str:
        task_id = str(uuid.uuid4())
        await self.queue.put((task_id, coro))
        return task_id

    def get_position(self, task_id: str) -> int:
        # Convert queue to list and search for the task ID
        queue_items = list(self.queue._queue)  # _queue is internal, but okay for read-only inspection
        for idx, (tid, _) in enumerate(queue_items):
            if tid == task_id:
                return idx + 1  # position in line (1-based)
        return 0  # not found or already processed

    async def _worker(self):
        while True:
            task_id, task_coro = await self.queue.get()
            try:
                await task_coro
            except Exception as e:
                print(f"[Analysis Queue] Task {task_id} failed: {e}")
            self.queue.task_done()

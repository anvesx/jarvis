"""One lock for the torch work on the Apple GPU (MPS) in this process.

PyTorch's MPS backend has one command stream per process, and two threads must not use it at once: a call from a
second thread can find a command buffer that the first thread just committed, and the process aborts. This happened
on 2026-10-22: the Parakeet keep-warm thread and the maya1 decoder ran together, and the speech service stopped with
"failed assertion _status < MTLCommandBufferStatusCommitted". So every torch call on MPS takes this lock.
MLX has its own command queues and does not take it. The lock is reentrant: a holder can call code that takes it.
"""
import threading

MPS = threading.RLock()

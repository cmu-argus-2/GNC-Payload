from contextlib import nullcontext
from multiprocessing import Array, Process, Queue
from multiprocessing.sharedctypes import SynchronizedArray
from time import perf_counter, sleep
from typing import Callable, List, ParamSpec, Tuple, TypeVar

import numpy as np
import psutil
from tqdm import tqdm

P = ParamSpec("P")
R = TypeVar("R")


class MemoryAwareProcessPool:
    """
    This class maintains a pool of worker processes to execute jobs in parallel. The number of jobs running in parallel
    will be dynamically adjusted to maintain the system memory usage within specified bounds. As a result, individual
    jobs may be abruptly terminated and restarted at a later time if the system memory usage gets too high.
    """

    def __init__(
        self,
        num_workers: int = 32,
        low_memory_usage_threshold: float = 80.0,
        high_memory_usage_threshold: float = 90.0,
        poll_interval: float = 2.0,
    ):
        """
        Initialize the MemoryAwareProcessPool.

        :param num_workers: The number of workers to use.
        :param low_memory_usage_threshold: The memory usage percentage at which it is considered safe to start new jobs.
        :param high_memory_usage_threshold: The memory usage percentage at which to start terminating worker processes
                                            to reclaim memory.
        :param poll_interval: The interval at which to poll the system memory usage, in seconds.
        """
        self.num_workers = num_workers
        self.low_memory_usage_threshold = low_memory_usage_threshold
        self.high_memory_usage_threshold = high_memory_usage_threshold
        self.poll_interval = poll_interval

        self.job_queue = Queue()
        self.result_queue = Queue()

        # shared array mapping worker processes to the job ID they are currently working on
        self.worker_job_ids: SynchronizedArray = Array("i", [-1] * self.num_workers)

        self.worker_processes = [
            self.make_new_worker_process(worker_id) for worker_id in range(self.num_workers)
        ]

        for worker_process in self.worker_processes:
            worker_process.start()

    @staticmethod
    def get_memory_usage_percentage() -> float:
        """
        Get the current system memory usage as a percentage.
        Note that swap memory is not included in this calculation.

        :return: The system memory usage as a percentage between 0 and 100.
        """
        return psutil.virtual_memory().percent

    @staticmethod
    def worker_main(
        worker_id: int,
        job_queue: Queue,
        result_queue: Queue,
        worker_job_ids: SynchronizedArray,
    ) -> None:
        """
        The main function for each worker process. The worker will continuously get jobs from the job queue and execute
        them until it receives a job with a job ID of -1 or is terminated by the parent process.

        :param worker_id: The ID of the worker process.
        :param job_queue: The queue to get jobs from.
        :param result_queue: The queue to put results into.
        :param worker_job_ids: The shared array to keep track of which job each worker is working on.
        """
        while True:
            job_id, func, args = job_queue.get()
            if job_id == -1:
                break

            # no need for a lock here since this is already atomic with respect to worker_job_ids.get_lock()
            worker_job_ids[worker_id] = job_id
            # termination is only possible after the above line, but before the lock is acquired below

            success = True
            try:
                result = func(*args)
            except Exception as e:
                result = e
                success = False

            # we need these two operations to be atomic to prevent the worker from being terminated after it has put
            # its result into the queue, but before it has updated its worker ID
            with worker_job_ids.get_lock():
                result_queue.put((job_id, success, result))
                worker_job_ids[worker_id] = -1

    def __enter__(self) -> "MemoryAwareProcessPool":
        """
        Enter the context manager.

        :return: This MemoryAwareProcessPool instance.
        """
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """
        Exit the context manager. This will terminate all worker processes gracefully.
        """
        for _ in range(self.num_workers):
            self.job_queue.put((-1, None, None))

        for worker_process in self.worker_processes:
            worker_process.join()

    def make_new_worker_process(self, worker_id: int) -> Process:
        """
        Create a new worker process. The worker process will need to be started separately.

        :param worker_id: The ID of the worker process.
        :return: The worker process.
        """
        return Process(
            target=self.worker_main,
            args=(
                worker_id,
                self.job_queue,
                self.result_queue,
                self.worker_job_ids,
            ),
            daemon=True,
        )

    def map(
        self,
        func: Callable[P, R],
        requests: List[P.args],
        display_progress_bar: bool = True,
        output_log_path: str | None = None,
    ) -> Tuple[np.ndarray, List[R | Exception]]:
        """
        Apply a function to each set of arguments from the provided list, in parallel using the worker pool.
        Note that this function will block until all requests have been completed.

        :param func: The function to apply. Must be pickleable.
        :param requests: A list where each element is an iterable of arguments to call the function with.
        :param display_progress_bar: Whether to display a progress bar.
        :param output_log_path: The path to write the output log to. If None, no log will be written.
        :return: A tuple containing:
                 - A numpy array of dtype bool indicating whether each request finished without an exception.
                 - A list of result objects or exceptions from each request.
        """
        num_jobs = len(requests)
        completed_requests = np.zeros(num_jobs, dtype=bool)
        successful_requests = np.zeros(num_jobs, dtype=bool)
        request_results = [None] * num_jobs

        next_job_id = 0
        # mapping for jobs that are currently being processed
        job_ids_to_request_ids: dict[int, int] = {}

        def terminate_process() -> bool:
            """
            Terminates the worker process with the largest job ID to free up memory.
            If no jobs are in progress, this function does nothing.

            The rationale is that the worker process with the largest job ID has been running for the least amount of
            time, so it is the most likely to be able to be terminated without losing much progress.

            :return: True if a process was terminated, False otherwise.
            """
            with self.worker_job_ids.get_lock():
                worker_id: int = np.argmax(list(self.worker_job_ids))
                job_id = self.worker_job_ids[worker_id]
                if job_id == -1:
                    return False

                self.worker_processes[worker_id].terminate()
                self.worker_processes[worker_id].join()

            # these operations have no potential for race conditions, so they don't need to be in the with block
            del job_ids_to_request_ids[job_id]
            self.worker_job_ids[worker_id] = -1
            self.worker_processes[worker_id] = self.make_new_worker_process(worker_id)
            self.worker_processes[worker_id].start()
            return True

        pbar = (
            tqdm(total=num_jobs, desc="0 jobs in progress", unit="request")
            if display_progress_bar
            else nullcontext()
        )
        output_log_file = (
            open(output_log_path, "w") if output_log_path is not None else nullcontext()
        )

        start_time = perf_counter()
        with pbar, output_log_file:
            if output_log_file is not None:
                output_log_file.write(
                    "Time elapsed,Memory usage percentage,Number of results received,Started job,Terminated process\n"
                )

            while not np.all(completed_requests):
                # for logging
                num_results_received = 0
                started_job = False
                terminated_process = False

                while not self.result_queue.empty():
                    job_id, success, result = self.result_queue.get()
                    request_id = job_ids_to_request_ids[job_id]

                    completed_requests[request_id] = True
                    successful_requests[request_id] = success
                    request_results[request_id] = result
                    del job_ids_to_request_ids[job_id]

                    num_results_received += 1
                    if display_progress_bar:
                        pbar.update(1)

                memory_usage_percentage = MemoryAwareProcessPool.get_memory_usage_percentage()

                if memory_usage_percentage > self.high_memory_usage_threshold:
                    terminated_process = terminate_process()
                elif (
                    memory_usage_percentage < self.low_memory_usage_threshold
                    and len(job_ids_to_request_ids) < self.num_workers
                ):
                    in_progress_requests = np.zeros(num_jobs, dtype=bool)
                    for request_id in job_ids_to_request_ids.values():
                        in_progress_requests[request_id] = True

                    eligible_requests = np.logical_and(~completed_requests, ~in_progress_requests)
                    if np.any(eligible_requests):
                        # get the first eligible request
                        request_id = np.argmax(eligible_requests)

                        self.job_queue.put((next_job_id, func, requests[request_id]))
                        job_ids_to_request_ids[next_job_id] = request_id
                        next_job_id += 1
                        started_job = True

                if display_progress_bar:
                    pbar.set_description(f"{len(job_ids_to_request_ids)} jobs in progress")

                if output_log_path is not None:
                    output_log_file.write(
                        f"{perf_counter() - start_time},{memory_usage_percentage},{num_results_received},"
                        f"{int(started_job)},{int(terminated_process)}\n"
                    )

                sleep(self.poll_interval)

        return successful_requests, request_results

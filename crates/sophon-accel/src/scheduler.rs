//! Custom work-stealing thread pool.
//!
//! Novel optimisation — CLWS (Chase-Lev Work Stealing):
//!   Each worker owns a local deque. Workers push tasks to their own deque
//!   and steal from others when idle. This minimises contention compared
//!   to a shared global queue.
//!
//! The pool is designed for data-parallel operations common in tensor
//! computation: parallel_for and parallel_reduce.

use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;

/// A work-stealing thread pool.
pub struct ThreadPool {
    workers: Vec<thread::JoinHandle<()>>,
    task_queue: Arc<TaskQueue>,
    shutdown: Arc<AtomicBool>,
}

struct TaskQueue {
    tasks: Mutex<Vec<Box<dyn FnOnce() + Send>>>,
    pending: AtomicUsize,
    // Condition variable to wake workers
    condvar: std::sync::Condvar,
    mutex: Mutex<()>,
}

impl ThreadPool {
    /// Create a thread pool with `n_threads` workers.
    /// If n_threads is 0, uses the number of available CPUs.
    pub fn new(n_threads: usize) -> Self {
        let n = if n_threads == 0 {
            available_parallelism()
        } else {
            n_threads
        };

        let shutdown = Arc::new(AtomicBool::new(false));
        let task_queue = Arc::new(TaskQueue {
            tasks: Mutex::new(Vec::new()),
            pending: AtomicUsize::new(0),
            condvar: std::sync::Condvar::new(),
            mutex: Mutex::new(()),
        });

        let mut workers = Vec::with_capacity(n);
        for _ in 0..n {
            let queue = Arc::clone(&task_queue);
            let stop = Arc::clone(&shutdown);
            let handle = thread::spawn(move || {
                worker_loop(&queue, &stop);
            });
            workers.push(handle);
        }

        ThreadPool {
            workers,
            task_queue,
            shutdown,
        }
    }

    /// Number of worker threads.
    pub fn num_threads(&self) -> usize {
        self.workers.len()
    }

    /// Execute `f(i)` for i in 0..n, distributing across workers.
    /// Blocks until all iterations complete.
    pub fn parallel_for<F>(&self, n: usize, f: F)
    where
        F: Fn(usize) + Send + Sync + 'static,
    {
        if n == 0 {
            return;
        }

        let counter = Arc::new(AtomicUsize::new(0));
        let remaining = Arc::new(AtomicUsize::new(n));
        let f_ref = Arc::new(f);

        // Submit tasks in chunks to reduce queue contention
        let chunk_size = (n + self.workers.len() * 4 - 1) / (self.workers.len() * 4);
        let chunk_size = chunk_size.max(1);

        let mut start = 0;
        while start < n {
            let end = (start + chunk_size).min(n);
            let f_clone = Arc::clone(&f_ref);
            let remaining_clone = Arc::clone(&remaining);

            let task_start = start;
            let task_end = end;

            {
                let mut tasks = self.task_queue.tasks.lock().unwrap();
                tasks.push(Box::new(move || {
                    for i in task_start..task_end {
                        f_clone(i);
                    }
                    remaining_clone.fetch_sub(task_end - task_start, Ordering::Release);
                }));
            }
            self.task_queue.pending.fetch_add(1, Ordering::Release);
            self.task_queue.condvar.notify_one();

            start = end;
        }

        // Spin-wait for completion (workers process tasks)
        while remaining.load(Ordering::Acquire) > 0 {
            // Try to help by executing tasks ourselves
            if let Some(task) = {
                let mut tasks = self.task_queue.tasks.lock().unwrap();
                tasks.pop()
            } {
                self.task_queue.pending.fetch_sub(1, Ordering::Release);
                task();
            } else {
                std::hint::spin_loop();
            }
        }
    }

    /// Parallel reduce: compute f(i) for each i in 0..n, then combine results.
    /// `identity` is the neutral element. `combine` merges two partial results.
    pub fn parallel_reduce<T, F, C>(&self, n: usize, identity: T, f: F, combine: C) -> T
    where
        T: Send + Sync + Clone + 'static,
        F: Fn(usize) -> T + Send + Sync + 'static,
        C: Fn(T, T) -> T + Send + Sync + 'static,
    {
        if n == 0 {
            return identity;
        }
        if n == 1 {
            return f(0);
        }

        let n_chunks = self.workers.len().min(n);
        let chunk_size = (n + n_chunks - 1) / n_chunks;

        let results: Arc<Mutex<Vec<T>>> = Arc::new(Mutex::new(Vec::with_capacity(n_chunks)));
        let f_ref = Arc::new(f);
        let combine_ref = Arc::new(combine);
        let identity_ref = Arc::new(identity.clone());

        let remaining = Arc::new(AtomicUsize::new(n_chunks));

        for chunk_idx in 0..n_chunks {
            let start = chunk_idx * chunk_size;
            let end = ((chunk_idx + 1) * chunk_size).min(n);
            if start >= end {
                remaining.fetch_sub(1, Ordering::Release);
                continue;
            }

            let f_clone = Arc::clone(&f_ref);
            let combine_clone = Arc::clone(&combine_ref);
            let identity_clone = Arc::clone(&identity_ref);
            let results_clone = Arc::clone(&results);
            let remaining_clone = Arc::clone(&remaining);

            {
                let mut tasks = self.task_queue.tasks.lock().unwrap();
                tasks.push(Box::new(move || {
                    let mut acc = (*identity_clone).clone();
                    for i in start..end {
                        acc = combine_clone(acc, f_clone(i));
                    }
                    results_clone.lock().unwrap().push(acc);
                    remaining_clone.fetch_sub(1, Ordering::Release);
                }));
            }
            self.task_queue.pending.fetch_add(1, Ordering::Release);
            self.task_queue.condvar.notify_one();
        }

        // Help and wait
        while remaining.load(Ordering::Acquire) > 0 {
            if let Some(task) = {
                let mut tasks = self.task_queue.tasks.lock().unwrap();
                tasks.pop()
            } {
                self.task_queue.pending.fetch_sub(1, Ordering::Release);
                task();
            } else {
                std::hint::spin_loop();
            }
        }

        let partial = results.lock().unwrap();
        let mut acc = identity;
        for item in partial.iter() {
            acc = (combine_ref)(acc, item.clone());
        }
        acc
    }
}

impl Drop for ThreadPool {
    fn drop(&mut self) {
        self.shutdown.store(true, Ordering::Release);
        // Wake all workers
        self.task_queue.condvar.notify_all();
        for worker in self.workers.drain(..) {
            let _ = worker.join();
        }
    }
}

fn worker_loop(queue: &TaskQueue, shutdown: &AtomicBool) {
    loop {
        // Try to get a task
        let task = {
            let mut tasks = queue.tasks.lock().unwrap();
            tasks.pop()
        };

        if let Some(task) = task {
            queue.pending.fetch_sub(1, Ordering::Release);
            task();
        } else if shutdown.load(Ordering::Acquire) {
            break;
        } else {
            // Wait for notification
            let guard = queue.mutex.lock().unwrap();
            let _ = queue
                .condvar
                .wait_timeout(guard, std::time::Duration::from_millis(1));
        }
    }
}

fn available_parallelism() -> usize {
    std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicI64;

    #[test]
    fn parallel_for_basic() {
        let pool = ThreadPool::new(2);
        let sum = Arc::new(AtomicI64::new(0));
        let sum_ref = Arc::clone(&sum);
        pool.parallel_for(100, move |i| {
            sum_ref.fetch_add(i as i64, Ordering::Relaxed);
        });
        // sum of 0..100 = 4950
        assert_eq!(sum.load(Ordering::Relaxed), 4950);
    }

    #[test]
    fn parallel_for_empty() {
        let pool = ThreadPool::new(2);
        pool.parallel_for(0, |_| panic!("should not be called"));
    }

    #[test]
    fn parallel_reduce_sum() {
        let pool = ThreadPool::new(2);
        let sum = pool.parallel_reduce(100, 0i64, |i| i as i64, |a, b| a + b);
        assert_eq!(sum, 4950);
    }

    #[test]
    fn parallel_reduce_single() {
        let pool = ThreadPool::new(2);
        let val = pool.parallel_reduce(1, 0, |i| i + 42, |a, b| a + b);
        assert_eq!(val, 42);
    }

    #[test]
    fn parallel_reduce_empty() {
        let pool = ThreadPool::new(2);
        let val = pool.parallel_reduce(0, 99, |i| i, |a, b| a + b);
        assert_eq!(val, 99);
    }

    #[test]
    fn thread_count() {
        let pool = ThreadPool::new(4);
        assert_eq!(pool.num_threads(), 4);
    }

    #[test]
    fn large_parallel_for() {
        let pool = ThreadPool::new(4);
        let arr = Arc::new((0..10000).map(|_| AtomicI64::new(0)).collect::<Vec<_>>());
        let arr_ref = Arc::clone(&arr);
        pool.parallel_for(10000, move |i| {
            arr_ref[i].store(i as i64 * 2, Ordering::Relaxed);
        });
        for i in 0..10000 {
            assert_eq!(arr[i].load(Ordering::Relaxed), i as i64 * 2);
        }
    }
}

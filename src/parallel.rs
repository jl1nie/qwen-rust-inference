//! CPU並列化（標準ライブラリだけ、依存追加なし）。
//!
//! 出力を連続したブロックに分け、ブロックごとに1スレッドが直列版と同じ
//! 関数で埋める。各出力要素は1スレッドが同じ順序で計算するので、結果は
//! スレッド数によらずビット単位で同一（加算順序を変えない）。
//!
//! decodeでは1トークンあたり約170回の射影がある。そのたびにスレッドを
//! 起動すると直列より遅くなった（実測：1スレッド20.9 → 24スレッド9.9 token/s）ので、
//! 起動済みのワーカーを使い回し、待機中は短時間spinしてから眠る。
//! 小さな仕事は呼び出し元で直列に実行する。
//! 同時に別の呼び出しが実行中（テストの並列実行、入れ子）なら直列に実行する。
//! `QWEN_THREADS=1` で直列に戻せる。
use std::{
    panic::{catch_unwind, resume_unwind, AssertUnwindSafe},
    sync::{
        atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering},
        Condvar, Mutex, OnceLock,
    },
    thread,
    time::{Duration, Instant},
};

/// Work (multiply-adds) per block below which an extra thread does not pay off.
const MIN_WORK_PER_BLOCK: usize = 1 << 15;
/// Waiting threads spin this long before sleeping: decode posts a job every
/// few tens of microseconds, and waking a sleeping thread costs about as much.
const SPIN: Duration = Duration::from_micros(100);
/// Spins until `done()` or SPIN elapsed; true if `done()` became true.
fn spin(done: impl Fn() -> bool) -> bool {
    let start = Instant::now();
    while !done() {
        if start.elapsed() > SPIN {
            return false;
        }
        std::hint::spin_loop();
    }
    true
}

/// Default cap on threads. Decode is memory-bandwidth bound: on a Ryzen 9
/// 9900X (WSL2) 2..8 threads all gave about 27 token/s vs 21 serial, while 24
/// (with SMT siblings spinning) fell to 15.
const DEFAULT_MAX_THREADS: usize = 8;
/// Worker count including the caller: `QWEN_THREADS`, else the available
/// cores up to DEFAULT_MAX_THREADS.
pub fn threads() -> usize {
    static N: OnceLock<usize> = OnceLock::new();
    *N.get_or_init(|| {
        std::env::var("QWEN_THREADS")
            .ok()
            .and_then(|v| v.parse().ok())
            .filter(|&n: &usize| n > 0)
            .unwrap_or_else(|| {
                thread::available_parallelism().map_or(1, |n| n.get().min(DEFAULT_MAX_THREADS))
            })
    })
}

/// Fills `out` in contiguous blocks: `f(first, block)` writes
/// `out[first..first + block.len()]`. `work` is the cost of one element (for
/// a GEMV, the row length); block boundaries are multiples of `align`.
pub fn for_blocks<T: Send>(
    out: &mut [T],
    work: usize,
    align: usize,
    f: impl Fn(usize, &mut [T]) + Sync,
) {
    let len = out.len();
    let total = len.saturating_mul(work.max(1));
    let n = threads().min(total / MIN_WORK_PER_BLOCK).max(1);
    let align = align.max(1);
    let block = len.div_ceil(n).div_ceil(align) * align;
    let Some(pool) = (n > 1 && block < len).then(pool).flatten() else {
        return f(0, out);
    };
    let Ok(_running) = pool.exclusive.try_lock() else {
        return f(0, out); // pool already busy: stay serial rather than wait
    };
    // Blocks are disjoint ranges of `out`; each index is claimed exactly once.
    let base = SendPtr(out.as_mut_ptr());
    let task = |i: usize| {
        let first = i * block;
        let size = block.min(len - first);
        // SAFETY: block i covers out[first..first+size], disjoint from every
        // other block, and `out` outlives the job (run() returns only after
        // all workers have left it).
        let chunk = unsafe { std::slice::from_raw_parts_mut(base.get().add(first), size) };
        f(first, chunk);
    };
    pool.run(len.div_ceil(block), &task);
}

struct SendPtr<T>(*mut T);
// Manual impls: derive would require T: Copy.
impl<T> Clone for SendPtr<T> {
    fn clone(&self) -> Self {
        *self
    }
}
impl<T> Copy for SendPtr<T> {}
// SAFETY: only used to hand disjoint sub-slices of one `&mut [T]` (T: Send) to workers.
unsafe impl<T: Send> Send for SendPtr<T> {}
unsafe impl<T: Send> Sync for SendPtr<T> {}
impl<T> SendPtr<T> {
    fn get(self) -> *mut T {
        self.0
    }
}

/// The task of the current job with its lifetime erased; valid while `run` waits.
type Task = *const (dyn Fn(usize) + Sync);
struct Job {
    generation: u64,
    task: Option<SendTask>,
    blocks: usize,
}
struct SendTask(Task);
// SAFETY: the pointee is Sync and outlives every use (see Pool::run).
unsafe impl Send for SendTask {}

struct Pool {
    exclusive: Mutex<()>,
    job: Mutex<Job>,
    wake: Condvar,
    left: Condvar,
    next: AtomicUsize,
    panicked: AtomicBool,
    /// Copy of `Job::generation` that workers can spin on without the lock.
    posted: AtomicU64,
    /// Workers that have entered the current job and not yet left it
    /// (incremented under the job lock, so none enters a withdrawn job).
    inside: AtomicUsize,
}

fn pool() -> Option<&'static Pool> {
    static POOL: OnceLock<Option<&'static Pool>> = OnceLock::new();
    *POOL.get_or_init(|| {
        let pool: &'static Pool = Box::leak(Box::new(Pool {
            exclusive: Mutex::new(()),
            job: Mutex::new(Job {
                generation: 0,
                task: None,
                blocks: 0,
            }),
            wake: Condvar::new(),
            left: Condvar::new(),
            next: AtomicUsize::new(0),
            panicked: AtomicBool::new(false),
            posted: AtomicU64::new(0),
            inside: AtomicUsize::new(0),
        }));
        for i in 1..threads() {
            let spawned = thread::Builder::new()
                .name(format!("qwen-worker-{i}"))
                .spawn(move || pool.work());
            if spawned.is_err() {
                return None; // fall back to serial execution
            }
        }
        Some(pool)
    })
}

impl Pool {
    /// Runs `task(0..blocks)` on the workers and the calling thread.
    fn run(&self, blocks: usize, task: &(dyn Fn(usize) + Sync)) {
        // SAFETY (lifetime erasure): `task` is only called through this
        // pointer while `inside > 0` or by this thread, and we return only
        // after the job is withdrawn and every worker has left it.
        let erased: Task = unsafe { std::mem::transmute(task) };
        self.next.store(0, Ordering::Relaxed);
        self.panicked.store(false, Ordering::Relaxed);
        {
            let mut job = self.job.lock().unwrap();
            job.generation += 1;
            job.task = Some(SendTask(erased));
            job.blocks = blocks;
            self.posted.store(job.generation, Ordering::Release);
        }
        self.wake.notify_all();
        self.claim(blocks, task);
        self.job.lock().unwrap().task = None; // late workers see no job
        if !spin(|| self.inside.load(Ordering::Acquire) == 0) {
            let mut job = self.job.lock().unwrap();
            while self.inside.load(Ordering::Acquire) > 0 {
                job = self.left.wait(job).unwrap();
            }
        }
        if self.panicked.load(Ordering::Relaxed) {
            resume_unwind(Box::new("parallel task panicked"));
        }
    }
    /// Takes block indices until none are left.
    fn claim(&self, blocks: usize, task: &(dyn Fn(usize) + Sync)) {
        loop {
            let i = self.next.fetch_add(1, Ordering::Relaxed);
            if i >= blocks {
                return;
            }
            if catch_unwind(AssertUnwindSafe(|| task(i))).is_err() {
                self.panicked.store(true, Ordering::Relaxed);
            }
        }
    }
    fn work(&self) {
        let mut seen = 0;
        loop {
            spin(|| self.posted.load(Ordering::Acquire) != seen);
            let (task, blocks) = {
                let mut job = self.job.lock().unwrap();
                while job.generation == seen || job.task.is_none() {
                    if job.generation != seen && job.task.is_none() {
                        seen = job.generation; // missed a finished job
                    }
                    job = self.wake.wait(job).unwrap();
                }
                seen = job.generation;
                self.inside.fetch_add(1, Ordering::AcqRel);
                (job.task.as_ref().unwrap().0, job.blocks)
            };
            // SAFETY: run() keeps the task alive until `inside` returns to 0.
            self.claim(blocks, unsafe { &*task });
            if self.inside.fetch_sub(1, Ordering::AcqRel) == 1 {
                // Taking the lock orders this notify after run()'s check.
                let _job = self.job.lock().unwrap();
                self.left.notify_all();
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn blocks_cover_every_element_once() {
        for len in [1, 63, 64, 1000, 4097] {
            let mut out = vec![0_usize; len];
            for _ in 0..3 {
                for_blocks(&mut out, MIN_WORK_PER_BLOCK, 64, |first, block| {
                    assert_eq!(first % 64, 0);
                    for (i, v) in block.iter_mut().enumerate() {
                        *v += first + i + 1;
                    }
                });
            }
            assert!(out.iter().enumerate().all(|(i, &v)| v == 3 * (i + 1)));
        }
    }
}

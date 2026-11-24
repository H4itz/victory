//! Memory readings for the log, and the optional commit floor that holds
//! back pipeline compiles while the system is short of commit.
//!
//! A world run on a 16 GiB machine ended with Windows out of commit: file
//! reads failed with `ERROR_NO_SYSTEM_RESOURCES` (0x800705AA) and then a
//! 1 MiB allocation of ours, while hundreds of pipelines were being
//! compiled. The frame-rate line said nothing about memory, so nothing
//! told which part grew. Each frame-rate line is now followed by one
//! `memory:` line: system commit (used, limit, free), this process's
//! private bytes and working set, the GPU's local (VRAM) and non-local
//! (system memory) usage against the budgets the driver gives, the host
//! copies of title memory ([`crate::shadow`]), and the pipelines still
//! being made. Below 2 GiB of free commit the line comes every second, as
//! a warning, so the last one before a failure is at most a second old.
//!
//! `VICTORY_PIPELINE_COMMIT_FLOOR_MB=<MiB>` (off by default): a pipeline
//! worker waits before its next compile while free commit is below the
//! floor. A driver compile of one of the world's big compute programs
//! takes hundreds of MiB for a moment, and several run at once.

use std::sync::OnceLock;
use std::sync::atomic::{AtomicU64, Ordering};

const MIB: u64 = 1 << 20;

/// System commit `(used, limit)` in bytes.
pub(crate) fn system_commit() -> Option<(u64, u64)> {
    use windows_sys::Win32::System::ProcessStatus::{
        K32GetPerformanceInfo, PERFORMANCE_INFORMATION,
    };
    let mut info = PERFORMANCE_INFORMATION {
        cb: size_of::<PERFORMANCE_INFORMATION>() as u32,
        ..Default::default()
    };
    // SAFETY: a writable, sized structure.
    if unsafe { K32GetPerformanceInfo(&mut info, info.cb) } == 0 {
        return None;
    }
    let page = info.PageSize as u64;
    Some((
        info.CommitTotal as u64 * page,
        info.CommitLimit as u64 * page,
    ))
}

/// Free system commit in bytes.
pub(crate) fn free_commit() -> Option<u64> {
    system_commit().map(|(used, limit)| limit.saturating_sub(used))
}

/// This process: `(private bytes, peak private bytes, working set)`.
fn process() -> Option<(u64, u64, u64)> {
    use windows_sys::Win32::System::ProcessStatus::{
        K32GetProcessMemoryInfo, PROCESS_MEMORY_COUNTERS, PROCESS_MEMORY_COUNTERS_EX,
    };
    use windows_sys::Win32::System::Threading::GetCurrentProcess;
    let mut c = PROCESS_MEMORY_COUNTERS_EX {
        cb: size_of::<PROCESS_MEMORY_COUNTERS_EX>() as u32,
        ..Default::default()
    };
    // SAFETY: our own process and a sized structure (the extended form
    // starts with the plain one).
    let ok = unsafe {
        K32GetProcessMemoryInfo(
            GetCurrentProcess(),
            (&raw mut c).cast::<PROCESS_MEMORY_COUNTERS>(),
            c.cb,
        )
    };
    (ok != 0).then_some((
        c.PrivateUsage as u64,
        c.PeakPagefileUsage as u64,
        c.WorkingSetSize as u64,
    ))
}

/// The host GPU's `(local usage, local budget, non-local usage, non-local
/// budget)` in bytes.
fn video() -> Option<(u64, u64, u64, u64)> {
    use windows::Win32::Graphics::Dxgi::*;
    let device = &crate::host::host()?.device;
    // SAFETY: plain DXGI queries on the host device's own adapter.
    unsafe {
        let luid = device.GetAdapterLuid();
        let factory: IDXGIFactory4 = CreateDXGIFactory2(DXGI_CREATE_FACTORY_FLAGS(0)).ok()?;
        let adapter: IDXGIAdapter3 = factory.EnumAdapterByLuid(luid).ok()?;
        let mut local = DXGI_QUERY_VIDEO_MEMORY_INFO::default();
        let mut shared = DXGI_QUERY_VIDEO_MEMORY_INFO::default();
        adapter
            .QueryVideoMemoryInfo(0, DXGI_MEMORY_SEGMENT_GROUP_LOCAL, &mut local)
            .ok()?;
        adapter
            .QueryVideoMemoryInfo(0, DXGI_MEMORY_SEGMENT_GROUP_NON_LOCAL, &mut shared)
            .ok()?;
        Some((
            local.CurrentUsage,
            local.Budget,
            shared.CurrentUsage,
            shared.Budget,
        ))
    }
}

/// Below this much free commit the memory line comes every second.
const LOW: u64 = 2048 * MIB;

/// Called with each frame-rate line (`periodic`) and about once a second
/// between them.
pub(crate) fn tick(periodic: bool) {
    let commit = system_commit();
    let low = commit.is_some_and(|(used, limit)| limit.saturating_sub(used) < LOW);
    if !periodic && !low {
        return;
    }
    let mib = |b: u64| b / MIB;
    let commit_text = match commit {
        Some((used, limit)) => format!(
            "system commit {} of {} MiB ({} MiB free)",
            mib(used),
            mib(limit),
            mib(limit.saturating_sub(used))
        ),
        None => "system commit unknown".into(),
    };
    let process_text = match process() {
        Some((private, peak, working)) => format!(
            "process private {} MiB (peak {}), working set {} MiB",
            mib(private),
            mib(peak),
            mib(working)
        ),
        None => "process unknown".into(),
    };
    let video_text = match video() {
        Some((lu, lb, nu, nb)) => format!(
            "GPU local {} of {} MiB budget, non-local {} of {} MiB budget",
            mib(lu),
            mib(lb),
            mib(nu),
            mib(nb)
        ),
        None => "GPU memory unknown".into(),
    };
    let (buffers, buffer_count, loose, loose_count, pool) = crate::shadow::host_copy_bytes();
    let text = format!(
        "memory: {commit_text}; {process_text}; {video_text}; host copies of title memory: \
         buffers {} MiB ({buffer_count}), loose {} MiB ({loose_count}), snapshot pool {} MiB; \
         pipelines being made {}",
        mib(buffers),
        mib(loose),
        mib(pool),
        crate::pipeline::making()
    );
    if low {
        tracing::warn!("{text} (low commit)");
    } else {
        tracing::info!("{text}");
    }
}

/// `VICTORY_PIPELINE_COMMIT_FLOOR_MB`, in bytes, if set.
fn floor() -> Option<u64> {
    static FLOOR: OnceLock<Option<u64>> = OnceLock::new();
    *FLOOR.get_or_init(|| {
        let v = std::env::var("VICTORY_PIPELINE_COMMIT_FLOOR_MB").ok()?;
        let mb: u64 = v.trim().parse().ok().filter(|&n| n > 0)?;
        tracing::warn!(
            floor_mib = mb,
            "VICTORY_PIPELINE_COMMIT_FLOOR_MB: pipeline compiles wait while free commit is below it"
        );
        Some(mb * MIB)
    })
}

/// Before a pipeline worker's next compile: waits while free commit is
/// below the floor (when one is set). Logs when a wait starts and every
/// 10 s of it.
pub(crate) fn wait_for_commit() {
    static WAITS: AtomicU64 = AtomicU64::new(0);
    let Some(floor) = floor() else { return };
    let started = std::time::Instant::now();
    let mut told = None::<std::time::Instant>;
    while let Some(free) = free_commit()
        && free < floor
    {
        if told.is_none_or(|t| t.elapsed() >= std::time::Duration::from_secs(10)) {
            told = Some(std::time::Instant::now());
            tracing::warn!(
                free_mib = free / MIB,
                floor_mib = floor / MIB,
                waited_s = started.elapsed().as_secs(),
                waits = WAITS.fetch_add(1, Ordering::Relaxed) + 1,
                "pipeline compile held back: free commit below the floor"
            );
        }
        std::thread::sleep(std::time::Duration::from_millis(250));
    }
}

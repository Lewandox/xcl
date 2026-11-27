//! `XCL_GPU_CULPRIT=1` (diagnostics, off by default): which dispatches a stalled queue has not
//! finished. Around every dispatch XCL records (title lists, console-recorded lists,
//! ExecuteIndirect of a translated compute pipeline) the list writes a running number into a
//! readback buffer with `WriteBufferImmediate`: `MARKER_IN` before (the work before it has
//! started), `MARKER_OUT` after (the work before it is done), one pair of words per PC list.
//! When the GPU-wait watchdog releases a wait, a list whose last number in is above its last
//! number out has work that started and is not done: those dispatches are logged with their
//! console compute shader (`fnv64` of the title's CS bytecode, the name of its file in the
//! translated-shader cache) and their groups.
//!
//! DRED breadcrumbs would say the same on a device removal, but they change the driver's
//! pipeline cache keys: every pipeline is built again, and a retail title's start then takes
//! minutes.

use std::collections::{HashMap, VecDeque};
use std::ffi::c_void;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{Mutex, OnceLock};
use windows::core::Interface;
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::DXGI_SAMPLE_DESC;

pub fn enabled() -> bool {
    static E: OnceLock<bool> = OnceLock::new();
    *E.get_or_init(|| std::env::var("XCL_GPU_CULPRIT").is_ok_and(|v| v.trim() == "1"))
}

/// Lists with a slot in the readback buffer (two words each).
const SLOTS: usize = 1 << 14;
/// Dispatches remembered by number (the oldest are forgotten).
const REMEMBERED: usize = 1 << 20;

/// The console CS of the compute pipelines the title made, by the pointer it holds.
static CS: Mutex<Option<HashMap<usize, u64>>> = Mutex::new(None);

/// A compute pipeline the title made: its console CS, for the log.
pub unsafe fn note_compute(pso: usize, desc: *const c_void) {
    if !enabled() || pso == 0 || !crate::gpu::shader::readable(desc as usize, std::mem::size_of::<D3D12_COMPUTE_PIPELINE_STATE_DESC>()) {
        return;
    }
    let d = std::ptr::read(desc as *const D3D12_COMPUTE_PIPELINE_STATE_DESC);
    let (p, n) = (d.CS.pShaderBytecode as usize, d.CS.BytecodeLength);
    if p == 0 || n == 0 || !crate::gpu::shader::readable(p, n) {
        return;
    }
    let h = crate::gpu::shader::fnv64(std::slice::from_raw_parts(p as *const u8, n));
    CS.lock().unwrap().get_or_insert_with(HashMap::new).insert(pso, h);
}

/// The title's pipeline, or the one it stands for (a background build).
fn cs_of(title_pso: usize, pso: usize) -> u64 {
    let g = CS.lock().unwrap();
    let Some(m) = g.as_ref() else { return 0 };
    [title_pso, pso, super::pso_async::resolve(title_pso)].iter().find_map(|p| m.get(p)).copied().unwrap_or(0)
}

struct Markers {
    _buf: ID3D12Resource,
    gpu: u64,
    cpu: *const u32,
    /// PC list → its slot
    slots: HashMap<usize, u32>,
    /// dispatch number → (list, what it was)
    what: HashMap<u32, (usize, String)>,
    order: VecDeque<u32>,
}
// SAFETY: `cpu` is the persistent map of `_buf`, read only under the mutex
unsafe impl Send for Markers {}

static MARKERS: Mutex<Option<Markers>> = Mutex::new(None);
static NEXT: AtomicU32 = AtomicU32::new(1);

unsafe fn make_markers(dev: &ID3D12Device) -> Option<Markers> {
    let heap = D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_READBACK, ..Default::default() };
    let desc = D3D12_RESOURCE_DESC {
        Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
        Width: (SLOTS * 8) as u64,
        Height: 1,
        DepthOrArraySize: 1,
        MipLevels: 1,
        SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
        Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
        ..Default::default()
    };
    let mut buf: Option<ID3D12Resource> = None;
    if let Err(e) = dev.CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &desc, D3D12_RESOURCE_STATE_COPY_DEST, None, &mut buf) {
        crate::xlog_warn!("GPU culprit: no readback buffer for the dispatch markers: {e}");
        return None;
    }
    let buf = buf?;
    let mut p: *mut c_void = std::ptr::null_mut();
    buf.Map(0, None, Some(&mut p)).ok()?;
    crate::xlog_info!("GPU culprit: dispatch markers on ({SLOTS} list slots)");
    Some(Markers { gpu: buf.GetGPUVirtualAddress(), _buf: buf, cpu: p as *const u32, slots: HashMap::new(), what: HashMap::new(), order: VecDeque::new() })
}

/// Before a dispatch on `list`: a marker in with a new number, which the caller hands to
/// `after_dispatch` once the dispatch is recorded (0: no marker). `groups` `[0, 0, n]`: an
/// indirect one of at most `n` dispatches.
pub unsafe fn before_dispatch(list: &ID3D12GraphicsCommandList, title_pso: usize, pso: usize, groups: [u32; 3]) -> u32 {
    if !enabled() {
        return 0;
    }
    let Ok(l2) = list.cast::<ID3D12GraphicsCommandList2>() else { return 0 };
    let mut g = MARKERS.lock().unwrap();
    if g.is_none() {
        let mut dev: Option<ID3D12Device> = None;
        if list.GetDevice(&mut dev).is_err() {
            return 0;
        }
        *g = make_markers(&dev.unwrap());
    }
    let Some(m) = g.as_mut() else { return 0 };
    let key = list.as_raw() as usize;
    let Some(slot) = slot_for(&mut m.slots, key, SLOTS) else { return 0 };
    let id = NEXT.fetch_add(1, Ordering::Relaxed);
    let size = match groups {
        [0, 0, n] => format!("indirect, up to {n} dispatches"),
        [x, y, z] => format!("groups {x}x{y}x{z}"),
    };
    m.what.insert(id, (key, format!("cs {:016x}, pipeline {pso:#x}, {size}", cs_of(title_pso, pso))));
    m.order.push_back(id);
    record(&mut RECORDED.lock().unwrap(), key, id);
    if m.order.len() > REMEMBERED {
        let old = m.order.pop_front().unwrap();
        m.what.remove(&old);
    }
    let p = D3D12_WRITEBUFFERIMMEDIATE_PARAMETER { Dest: m.gpu + slot as u64 * 8, Value: id };
    l2.WriteBufferImmediate(1, &p, Some(&D3D12_WRITEBUFFERIMMEDIATE_MODE_MARKER_IN));
    id
}

/// The dispatches numbered on each PC list since its last submit (`submitted` takes them).
static RECORDED: Mutex<Option<HashMap<usize, Vec<u32>>>> = Mutex::new(None);
/// The newest submits of title queues: (queue, its value after the submit, each list with the
/// dispatches numbered on it). A stalled queue's first value not done names the lists the GPU
/// is in, and the markers alone do not: a list whose dispatches are all done can still be the
/// one that does not end (a copy, a clear, a barrier that waits for an earlier shader).
static SUBMITS: Mutex<VecDeque<(usize, u64, Vec<(usize, Vec<u32>)>)>> = Mutex::new(VecDeque::new());
/// Submits remembered, and the dispatches remembered of one list's recording.
const SUBMITS_KEPT: usize = 128;
const LIST_DISPATCHES_KEPT: usize = 1024;

fn record(recorded: &mut Option<HashMap<usize, Vec<u32>>>, list: usize, id: u32) {
    let v = recorded.get_or_insert_with(HashMap::new).entry(list).or_default();
    if v.len() < LIST_DISPATCHES_KEPT {
        v.push(id);
    }
}

/// The dispatches numbered on `lists` since their last submit, taken.
fn take_recorded(recorded: &mut Option<HashMap<usize, Vec<u32>>>, lists: &[usize]) -> Vec<(usize, Vec<u32>)> {
    let Some(m) = recorded.as_mut() else { return lists.iter().map(|&l| (l, Vec::new())).collect() };
    lists.iter().map(|&l| (l, m.remove(&l).unwrap_or_default())).collect()
}

/// A title queue's submit of `lists` (PC lists), after which the queue's fence has `value`.
pub fn submitted(queue: usize, value: u64, lists: &[usize]) {
    if !enabled() {
        return;
    }
    let taken = take_recorded(&mut RECORDED.lock().unwrap(), lists);
    let mut g = SUBMITS.lock().unwrap();
    g.push_back((queue, value, taken));
    if g.len() > SUBMITS_KEPT {
        g.pop_front();
    }
}

/// What a list's pair of words says of its dispatch `id`: the list's newest number in is at or
/// past it once the work before it has started, its newest number out once it is done.
fn state_of(id: u32, in_: u32, out: u32) -> &'static str {
    if out >= id {
        "done"
    } else if in_ >= id {
        "started"
    } else {
        "not started"
    }
}

/// Where in title memory the arguments of the indirect dispatches numbered so are (the newest
/// 65 536): a hung one is logged with the group counts there.
static ARGS: Mutex<Option<HashMap<u32, usize>>> = Mutex::new(None);

/// The indirect dispatch `id` (of `before_dispatch`) takes its group counts from `at` in title
/// memory.
pub fn indirect_args(id: u32, at: usize) {
    if id == 0 {
        return;
    }
    let mut g = ARGS.lock().unwrap();
    let m = g.get_or_insert_with(HashMap::new);
    if m.len() >= 1 << 16 {
        m.clear();
    }
    m.insert(id, at);
}

/// ` ; its arguments in title memory now: x×y×z groups` for the dispatch `id`, when known. What
/// the GPU read can differ: these are the bytes the CPU sees (a count another dispatch wrote may
/// not be back yet).
fn args_now(id: u32) -> String {
    let Some(at) = ARGS.lock().unwrap().as_ref().and_then(|m| m.get(&id).copied()) else { return String::new() };
    if !crate::gpu::shader::readable(at, 12) {
        return format!("; its arguments at {at:#x} cannot be read");
    }
    // SAFETY: 12 readable bytes, checked above
    let [x, y, z] = unsafe { (at as *const [u32; 3]).read_unaligned() };
    format!("; its arguments in title memory ({at:#x}) now: {x}x{y}x{z} groups")
}

/// After the dispatch: a marker out with its number.
pub unsafe fn after_dispatch(list: &ID3D12GraphicsCommandList, id: u32) {
    if id == 0 {
        return;
    }
    let Ok(l2) = list.cast::<ID3D12GraphicsCommandList2>() else { return };
    let g = MARKERS.lock().unwrap();
    let Some(m) = g.as_ref() else { return };
    let Some(&slot) = m.slots.get(&(list.as_raw() as usize)) else { return };
    let p = D3D12_WRITEBUFFERIMMEDIATE_PARAMETER { Dest: m.gpu + slot as u64 * 8 + 4, Value: id };
    l2.WriteBufferImmediate(1, &p, Some(&D3D12_WRITEBUFFERIMMEDIATE_MODE_MARKER_OUT));
}

/// The slot of `list`: its own, or the next free one; none once `cap` lists have one (the list is
/// then not recorded, so every recorded slot stays inside the readback buffer).
fn slot_for(slots: &mut HashMap<usize, u32>, list: usize, cap: usize) -> Option<u32> {
    if let Some(&s) = slots.get(&list) {
        return Some(s);
    }
    let n = slots.len();
    (n < cap).then(|| {
        slots.insert(list, n as u32);
        n as u32
    })
}

/// The lists with work started and not done: (list, last number in, last number out), from
/// each slot's pair of words. A marker in only says that the work before it has started, so a
/// list counts when its number in is above its number out (in and out written out of order
/// otherwise look open).
fn open_lists(slots: &HashMap<usize, u32>, words: impl Fn(u32) -> (u32, u32)) -> Vec<(usize, u32, u32)> {
    let mut open: Vec<(usize, u32, u32)> = slots
        .iter()
        .filter_map(|(&list, &slot)| {
            let (i, o) = words(slot);
            (i > o).then_some((list, i, o))
        })
        .collect();
    open.sort_by_key(|&(_, i, _)| i);
    open
}

/// The dispatches of `list` numbered after `out` up to `in_` that are still remembered.
fn not_done(what: &HashMap<u32, (usize, String)>, list: usize, in_: u32, out: u32) -> Vec<(u32, &str)> {
    ((out + 1)..=in_).filter_map(|id| what.get(&id).filter(|(l, _)| *l == list).map(|(_, s)| (id, s.as_str()))).collect()
}

/// The watchdog released GPU waits: log the dispatches started and not done (once per burst,
/// 4 s apart, at most 8 times).
pub fn on_watchdog_release() {
    static BURSTS: AtomicU32 = AtomicU32::new(0);
    static LAST: Mutex<Option<std::time::Instant>> = Mutex::new(None);
    if !enabled() {
        return;
    }
    {
        let mut l = LAST.lock().unwrap();
        if l.is_some_and(|t| t.elapsed() < std::time::Duration::from_secs(4)) {
            return;
        }
        *l = Some(std::time::Instant::now());
    }
    if BURSTS.fetch_add(1, Ordering::Relaxed) >= 8 {
        return;
    }
    report();
}

/// The device was removed: the same report, once. A driver that gives up on a hung GPU after
/// 2 s does so before the watchdog's 3 s without progress, and the report never came.
pub fn on_device_removed() {
    static DONE: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
    if enabled() && !DONE.swap(true, Ordering::Relaxed) {
        report();
    }
}

fn report() {
    let g = MARKERS.lock().unwrap();
    let Some(m) = g.as_ref() else { return };
    // SAFETY: slots index the mapped buffer of SLOTS pairs
    let open = open_lists(&m.slots, |s| unsafe { (std::ptr::read_volatile(m.cpu.add(s as usize * 2)), std::ptr::read_volatile(m.cpu.add(s as usize * 2 + 1))) });
    crate::xlog_warn!("GPU culprit: {} of {} lists have dispatches started and not done", open.len(), m.slots.len());
    for (list, i, o) in open.iter().take(16) {
        let left = not_done(&m.what, *list, *i, *o);
        crate::xlog_warn!("GPU culprit: list {list:#x}: marker in #{i}, last marker out #{o}; {} of its dispatches not known done:", left.len());
        for (id, s) in left.iter().take(24) {
            crate::xlog_warn!("GPU culprit:   #{id}: {s}{}", args_now(*id));
        }
    }
    // the newest submits, with what was on their lists (the first report and a device removal)
    static SUBMIT_REPORTS: AtomicU32 = AtomicU32::new(0);
    if SUBMIT_REPORTS.fetch_add(1, Ordering::Relaxed) >= 2 {
        return;
    }
    let submits = SUBMITS.lock().unwrap();
    crate::xlog_warn!("GPU culprit: the newest {} submits of title queues (queue, value, lists with their dispatches):", submits.len());
    for (queue, value, lists) in submits.iter() {
        let total: usize = lists.iter().map(|(_, d)| d.len()).sum();
        crate::xlog_warn!("GPU culprit:  queue {queue:#x} value {value}: {} lists, {total} dispatches", lists.len());
        for (list, ids) in lists.iter().filter(|(_, d)| !d.is_empty()) {
            // SAFETY: slots index the mapped buffer of SLOTS pairs
            let (i, o) = m.slots.get(list).map_or((0, 0), |&s| unsafe { (std::ptr::read_volatile(m.cpu.add(s as usize * 2)), std::ptr::read_volatile(m.cpu.add(s as usize * 2 + 1))) });
            let shown: Vec<String> = ids.iter().take(48).map(|id| format!("#{id} {} ({})", m.what.get(id).map_or("forgotten", |w| w.1.split(',').next().unwrap_or("")), state_of(*id, i, o))).collect();
            crate::xlog_warn!("GPU culprit:    list {list:#x} (in #{i}, out #{o}): {}{}", shown.join(", "), if ids.len() > 48 { ", …" } else { "" });
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A submit takes the dispatches numbered on its lists since their last submit, and a
    /// list's words say which of them the GPU has started and finished.
    #[test]
    fn a_submit_takes_its_lists_dispatches_and_the_words_give_their_state() {
        let mut rec = None;
        record(&mut rec, 0x10, 5);
        record(&mut rec, 0x20, 6);
        record(&mut rec, 0x10, 7);
        assert_eq!(take_recorded(&mut rec, &[0x10, 0x30]), vec![(0x10, vec![5, 7]), (0x30, vec![])]);
        // taken once: the list's next recording starts empty, the other list keeps its own
        assert_eq!(take_recorded(&mut rec, &[0x10, 0x20]), vec![(0x10, vec![]), (0x20, vec![6])]);
        assert_eq!(take_recorded(&mut None, &[0x10]), vec![(0x10, vec![])]);
        // in #7, out #5: 5 is done, 7 has started, 9 has not
        assert_eq!((state_of(5, 7, 5), state_of(7, 7, 5), state_of(9, 7, 5)), ("done", "started", "not started"));
        // a recording never holds more than its cap
        let mut rec = None;
        for id in 0..(LIST_DISPATCHES_KEPT as u32 + 10) {
            record(&mut rec, 1, id);
        }
        assert_eq!(take_recorded(&mut rec, &[1])[0].1.len(), LIST_DISPATCHES_KEPT);
    }

    /// Only a list whose number in is above its number out has work in flight, and the
    /// dispatches logged for it are its own, after the out.
    #[test]
    fn a_list_with_a_marker_in_above_its_marker_out_is_open() {
        let slots: HashMap<usize, u32> = [(0xA, 0), (0xB, 1), (0xC, 2), (0xD, 3)].into_iter().collect();
        // A: done; B: 12 started, 10 done; C: in and out written out of order; D: never ran
        let words = |s: u32| [(9, 9), (12, 10), (20, 21), (0, 0)][s as usize];
        assert_eq!(open_lists(&slots, words), vec![(0xB, 12, 10)]);

        let what: HashMap<u32, (usize, String)> =
            [(10, (0xB, "b10".to_string())), (11, (0xA, "a11".to_string())), (12, (0xB, "b12".to_string()))].into_iter().collect();
        assert_eq!(not_done(&what, 0xB, 12, 10), vec![(12, "b12")], "11 belongs to another list");
    }

    /// Once every slot is taken, a new list gets none and is not recorded, and the lists that
    /// have one keep it: no slot past the readback buffer is ever read.
    #[test]
    fn a_list_past_the_last_slot_gets_none() {
        let mut slots = HashMap::new();
        assert_eq!(slot_for(&mut slots, 0xA, 2), Some(0));
        assert_eq!(slot_for(&mut slots, 0xB, 2), Some(1));
        assert_eq!(slot_for(&mut slots, 0xC, 2), None);
        assert_eq!(slots.len(), 2, "the list without a slot is not recorded");
        assert_eq!(slot_for(&mut slots, 0xA, 2), Some(0));
        assert!(slots.values().all(|&s| (s as usize) < 2));
    }

    /// The markers reach the readback buffer: a list run to its end leaves its numbers in and
    /// out equal, so nothing is reported for it.
    #[test]
    fn gpu_markers_of_a_finished_list_match() {
        use windows::Win32::Graphics::Direct3D::D3D_FEATURE_LEVEL_11_0;
        unsafe {
            let mut dev: Option<ID3D12Device> = None;
            if D3D12CreateDevice(None, D3D_FEATURE_LEVEL_11_0, &mut dev).is_err() {
                crate::gpu::recomp::runner::skip_test("no GPU");
                return;
            }
            let dev = dev.unwrap();
            let Some(m) = make_markers(&dev) else { panic!("readback buffer") };
            let ty = D3D12_COMMAND_LIST_TYPE_COMPUTE;
            let queue: ID3D12CommandQueue = dev.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC { Type: ty, ..Default::default() }).unwrap();
            let alloc: ID3D12CommandAllocator = dev.CreateCommandAllocator(ty).unwrap();
            let list: ID3D12GraphicsCommandList = dev.CreateCommandList(0, ty, &alloc, None).unwrap();
            let l2: ID3D12GraphicsCommandList2 = list.cast().unwrap();
            for (i, mode) in [(0u64, D3D12_WRITEBUFFERIMMEDIATE_MODE_MARKER_IN), (4, D3D12_WRITEBUFFERIMMEDIATE_MODE_MARKER_OUT)] {
                let p = D3D12_WRITEBUFFERIMMEDIATE_PARAMETER { Dest: m.gpu + 8 + i, Value: 77 };
                l2.WriteBufferImmediate(1, &p, Some(&mode));
            }
            list.Close().unwrap();
            queue.ExecuteCommandLists(&[Some(list.cast().unwrap())]);
            let fence: ID3D12Fence = dev.CreateFence(0, D3D12_FENCE_FLAG_NONE).unwrap();
            queue.Signal(&fence, 1).unwrap();
            let t = std::time::Instant::now();
            while fence.GetCompletedValue() < 1 {
                assert!(t.elapsed().as_secs() < 10, "the list did not finish");
                std::thread::yield_now();
            }
            let words = |s: u32| (std::ptr::read_volatile(m.cpu.add(s as usize * 2)), std::ptr::read_volatile(m.cpu.add(s as usize * 2 + 1)));
            assert_eq!(words(1), (77, 77));
            let slots: HashMap<usize, u32> = [(list.as_raw() as usize, 1)].into_iter().collect();
            assert!(open_lists(&slots, words).is_empty());
        }
    }
}

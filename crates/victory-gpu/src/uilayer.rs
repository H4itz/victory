//! The UI layer drawn over the presented picture (`VICTORY_COMPOSE_UI`).
//!
//! The console shows several planes at once: the game's picture and a UI
//! plane. On the Series S path the UI plane (menus, loading screens, the
//! account picker, the HUD) is a 3840x2160 8-bit target while the game
//! renders at 1280x720, and the present shows only the first plane's first
//! resource, so those screens are black. With the switch the frame's UI
//! layer is blended over the first plane ([`crate::compose`]).
//!
//! shortcut: the UI layer is found by shape, not from the present's plane
//! list (its layout past the first plane is not known yet): the largest
//! 8-bit colour target the frame rendered to, 16:9, single-sample and at
//! least 2560 wide. It is taken as premultiplied
//! (`VICTORY_COMPOSE_UI=straight` for straight alpha). A frame that draws
//! no UI layer shows the last one again: the title redraws it only when it
//! changes.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

use crate::object::This;
use crate::resource::HostTexture;

/// `Some(straight alpha)` when composing.
pub(crate) fn mode() -> Option<bool> {
    static MODE: OnceLock<Option<bool>> = OnceLock::new();
    *MODE.get_or_init(|| {
        let v = std::env::var("VICTORY_COMPOSE_UI").ok()?;
        let straight = match v.trim().to_ascii_lowercase().as_str() {
            "1" | "on" | "true" | "yes" | "premultiplied" => false,
            "straight" => true,
            _ => return None,
        };
        tracing::warn!(straight, "VICTORY_COMPOSE_UI: the UI layer is drawn over the presented picture");
        Some(straight)
    })
}

/// Targets seen this frame: whether each is UI-layer shaped, and if so its
/// host copy and area. Keyed by the title resource.
#[allow(clippy::type_complexity)]
static SEEN: Mutex<Vec<(usize, Option<(Arc<HostTexture>, u64)>)>> = Mutex::new(Vec::new());

/// The UI layer shown last.
static LAST: Mutex<Option<Arc<HostTexture>>> = Mutex::new(None);

/// Whether this frame drew a UI layer (cheap check before locking).
static DRAWN: AtomicBool = AtomicBool::new(false);

/// Notes a render target the frame being built draws to.
pub(crate) fn note(resource: This) {
    if resource.is_null() || mode().is_none() {
        return;
    }
    let key = resource as usize;
    let mut seen = SEEN.lock().unwrap_or_else(|e| e.into_inner());
    if seen.iter().any(|(r, _)| *r == key) {
        return;
    }
    let shaped = crate::resource::host_copy(resource).and_then(|copy| {
        // SAFETY: a live resource.
        let d = unsafe { copy.texture.GetDesc() };
        (d.Width >= 2560
            && d.Width * 9 == u64::from(d.Height) * 16
            && d.SampleDesc.Count == 1
            && matches!(d.Format.0, 27..=29 | 87 | 90 | 91))
            .then(|| (copy, d.Width * u64::from(d.Height)))
    });
    if shaped.is_some() {
        DRAWN.store(true, Ordering::Relaxed);
    }
    seen.push((key, shaped));
}

/// At a present: the UI layer to blend over the picture, if any; starts
/// the next frame's list.
pub(crate) fn take() -> Option<Arc<HostTexture>> {
    mode()?;
    let mut seen = SEEN.lock().unwrap_or_else(|e| e.into_inner());
    let mut last = LAST.lock().unwrap_or_else(|e| e.into_inner());
    if DRAWN.swap(false, Ordering::Relaxed)
        && let Some((copy, _)) = seen
            .iter()
            .filter_map(|(_, s)| s.as_ref())
            .max_by_key(|(_, area)| *area)
    {
        *last = Some(copy.clone());
    }
    seen.clear();
    last.clone()
}

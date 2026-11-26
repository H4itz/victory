//! A stand-in for an upscaler that does not run.
//!
//! On Series S, Grand Theft Auto VI renders the scene at 1280x720 and AMD's
//! FSR 4 (`amd_fidelityfx_upscaler_xbox.dll`) brings it to 1920x1080, into a
//! shared-exponent (R9G9B9E5) texture that bloom, exposure and the final
//! composite read. Its Xbox backend wants D3D12.X methods we do not have
//! (`DeserializeComputePipelineStateX`, a pipeline state method at 0x58), so
//! none of its passes reach the GPU and that texture stays black: exposure
//! measured nothing and went to its limit, and the picture came out in
//! saturated primaries and black (RTX 3050 frame dump: the 1920x1080 target
//! all zeros, never written in the frame).
//!
//! Here, right before the first program that reads it (the bloom
//! downsample), the target gets the scene as the luminance pass read it,
//! scaled up bilinearly. No sharpening, which FSR does.
//!
//! FSR 4 also evens the picture out over frames: the title jitters its view
//! by a fraction of a pixel each frame for it, and dithers hair, shadows and
//! reflections, and its light leaves blocks that change from frame to frame.
//! Without it, each frame showed all of that (RTX 3050, FSR 4's 27 programs
//! left out: hair as grain, walls speckled). So the stand-in keeps what it
//! made the frame before and mixes it in (`VICTORY_UPSCALE_HISTORY=<share>`,
//! 0.7 unless set; 0: none), bounded by the scene's values around the point,
//! the 3x3 texels it is between: what moved falls outside them and is the
//! new frame's. There are no motion vectors here, so a moving picture is
//! the new frame's alone, as before; one that stands still settles.
//!
//! Only where the target is a shared-exponent stand-in
//! ([`crate::resource::stands_in`]) larger than the scene: the upscaler's
//! own output (Series S); the Series X profile does not take this path. And
//! only while no compute program has written the target through unordered
//! access since the scene was read: an upscaler that runs keeps its output.
//! `VICTORY_UPSCALE=0` turns it off.

use std::sync::atomic::{AtomicU32, AtomicUsize, Ordering};
use windows::Win32::Graphics::Direct3D::Fxc::D3DCompile;
use windows::Win32::Graphics::Direct3D::ID3DBlob;
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::*;
use windows::core::{PCSTR, s};

use crate::host::HostList;
use crate::object::This;

/// The program that reads the scene for exposure: where the scene is taken.
pub(crate) const SCENE_READER: &[u8] = b"PS_AlbedoFreeLuminanceSourceDownsample";
/// Its scene view: root parameter and place in its table.
pub(crate) const SCENE: (u8, u32) = (0, 0);
/// The first programs that read the upscaled scene (the bloom downsample,
/// not its pass over the TAA history).
pub(crate) const UPSCALED_READER: &[u8] = b"CS_BloomDownsampleKaris";
pub(crate) const NOT_UPSCALED_READER: &[u8] = b"TAAHistory";
/// Their view of the upscaled scene.
pub(crate) const UPSCALED: (u8, u32) = (1, 8);

/// How much of the frame before is kept (`VICTORY_UPSCALE_HISTORY`).
fn history_share() -> f32 {
    static SHARE: std::sync::OnceLock<f32> = std::sync::OnceLock::new();
    *SHARE.get_or_init(|| parse_share(&std::env::var("VICTORY_UPSCALE_HISTORY").unwrap_or_default()))
}

fn parse_share(set: &str) -> f32 {
    match set.trim() {
        "" => 0.7,
        v => v.parse::<f32>().ok().filter(|v| v.is_finite()).map_or(0.7, |v| v.clamp(0.0, 0.95)),
    }
}

/// The scene's title texture, as last read by [`SCENE_READER`].
static LAST_SCENE: AtomicUsize = AtomicUsize::new(0);
/// The target last filled, and whether a program wrote it through
/// unordered access since the scene was last read (an upscaler ran).
static LAST_TARGET: AtomicUsize = AtomicUsize::new(0);
static WRITTEN: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// A program writes `resource` through unordered access (`draw.rs`
/// `readable`): one compare per view.
pub(crate) fn note_unordered(resource: This) {
    if resource as usize == LAST_TARGET.load(Ordering::Relaxed) && resource as usize != 0 {
        WRITTEN.store(true, Ordering::Relaxed);
    }
}

pub(crate) fn on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_UPSCALE").is_none_or(|v| v != "0"))
}

/// [`SCENE_READER`] reads `scene`.
pub(crate) fn note_scene(scene: This) {
    LAST_SCENE.store(scene as usize, Ordering::Relaxed);
    WRITTEN.store(false, Ordering::Relaxed);
}

const SOURCE: &str = r#"
#define RS "DescriptorTable(SRV(t0, numDescriptors=2), visibility=SHADER_VISIBILITY_PIXEL), RootConstants(num32BitConstants=1, b0, visibility=SHADER_VISIBILITY_PIXEL), StaticSampler(s0, filter=FILTER_MIN_MAG_MIP_LINEAR, addressU=TEXTURE_ADDRESS_CLAMP, addressV=TEXTURE_ADDRESS_CLAMP, visibility=SHADER_VISIBILITY_PIXEL)"
Texture2D<float4> scene : register(t0);
Texture2D<float4> history : register(t1);
SamplerState smooth_ : register(s0);
cbuffer Set : register(b0) { float share; };
struct V { float4 at : SV_Position; float2 uv : TEXCOORD0; };
[RootSignature(RS)]
V vs(uint id : SV_VertexID) {
    V v;
    v.uv = float2((id << 1) & 2, id & 2);
    v.at = float4(v.uv * float2(2, -2) + float2(-1, 1), 0, 1);
    return v;
}
[RootSignature(RS)]
float4 ps(V v) : SV_Target {
    float3 now = scene.SampleLevel(smooth_, v.uv, 0).rgb;
    if (share <= 0.0) return float4(now, 1);
    uint w, h;
    scene.GetDimensions(w, h);
    float2 size = float2(w, h);
    // The texels around: their centres, read through the linear sampler.
    float2 c = (floor(v.uv * size) + 0.5) / size;
    float3 lo = now, hi = now;
    [unroll] for (int y = -1; y <= 1; y++) {
        [unroll] for (int x = -1; x <= 1; x++) {
            float3 t = scene.SampleLevel(smooth_, c + float2(x, y) / size, 0).rgb;
            lo = min(lo, t);
            hi = max(hi, t);
        }
    }
    float3 was = history.Load(int3(int2(v.at.xy), 0)).rgb;
    if (any(isnan(was)) || any(isinf(was))) was = now;
    return float4(lerp(now, clamp(was, lo, hi), share), 1);
}
"#;

/// Views, used in turn two at a time (the scene, what was made the frame
/// before): more than the upscales of every list in flight.
const SLOTS: u32 = 64;

/// What the stand-in made the frame before, for the next to mix in: a
/// texture of the target's size and format, kept readable by shaders
/// between upscales (each upscale copies into it and gives it back so,
/// whatever order lists are recorded in).
struct History {
    texture: ID3D12Resource,
    size: (u64, u32),
    format: i32,
}

struct Upscaler {
    root: ID3D12RootSignature,
    vs: ID3DBlob,
    ps: ID3DBlob,
    pipelines: std::sync::Mutex<Vec<(i32, ID3D12PipelineState)>>,
    views: ID3D12DescriptorHeap,
    step: u32,
    next: AtomicU32,
    history: std::sync::Mutex<Option<History>>,
}
// SAFETY: D3D12 objects are free-threaded; blobs are only read.
unsafe impl Send for Upscaler {}
// SAFETY: as above.
unsafe impl Sync for Upscaler {}

fn compile(entry: PCSTR, target: PCSTR) -> windows::core::Result<ID3DBlob> {
    let mut code: Option<ID3DBlob> = None;
    // SAFETY: the source and names live through the call; the out pointer is
    // ours.
    unsafe {
        D3DCompile(
            SOURCE.as_ptr().cast(),
            SOURCE.len(),
            s!("upscale"),
            None,
            None,
            entry,
            target,
            0,
            0,
            &mut code,
            None,
        )?;
    }
    code.ok_or_else(windows::core::Error::empty)
}

fn bytes(b: &ID3DBlob) -> &[u8] {
    // SAFETY: the blob's own bytes, alive as long as it.
    unsafe { std::slice::from_raw_parts(b.GetBufferPointer().cast::<u8>(), b.GetBufferSize()) }
}

fn make(device: &ID3D12Device) -> windows::core::Result<Upscaler> {
    let (vs, ps) = (compile(s!("vs"), s!("vs_5_1"))?, compile(s!("ps"), s!("ps_5_1"))?);
    // SAFETY: plain object creation from the compiler's blobs.
    unsafe {
        let root: ID3D12RootSignature = device.CreateRootSignature(0, bytes(&vs))?;
        let views = device.CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
            Type: D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
            NumDescriptors: SLOTS,
            Flags: D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE,
            NodeMask: 0,
        })?;
        Ok(Upscaler {
            root,
            vs,
            ps,
            pipelines: std::sync::Mutex::new(Vec::new()),
            views,
            step: device.GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV),
            next: AtomicU32::new(0),
            history: std::sync::Mutex::new(None),
        })
    }
}

impl Upscaler {
    /// The texture of what was made the frame before for a target like
    /// `like`, and whether it holds nothing yet (made now, or never copied
    /// into). None if it cannot be made.
    fn history(&self, device: &ID3D12Device, like: &D3D12_RESOURCE_DESC) -> Option<(ID3D12Resource, bool)> {
        let mut kept = self.history.lock().unwrap_or_else(|e| e.into_inner());
        let size = (like.Width, like.Height);
        if let Some(h) = kept.as_ref().filter(|h| h.size == size && h.format == like.Format.0) {
            return Some((h.texture.clone(), false));
        }
        let desc = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
            Width: like.Width,
            Height: like.Height,
            DepthOrArraySize: 1,
            MipLevels: 1,
            Format: like.Format,
            SampleDesc: DXGI_SAMPLE_DESC {
                Count: 1,
                Quality: 0,
            },
            ..Default::default()
        };
        let mut texture: Option<ID3D12Resource> = None;
        // SAFETY: a complete description; the out pointer is ours.
        let made = unsafe {
            device.CreateCommittedResource(
                &D3D12_HEAP_PROPERTIES {
                    Type: D3D12_HEAP_TYPE_DEFAULT,
                    ..Default::default()
                },
                D3D12_HEAP_FLAG_NONE,
                &desc,
                D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE,
                None,
                &mut texture,
            )
        };
        let Some(texture) = made.ok().and(texture) else {
            static ONCE: std::sync::Once = std::sync::Once::new();
            ONCE.call_once(|| tracing::warn!("the stand-in upscale keeps no frame before: no texture for it"));
            return None;
        };
        tracing::info!(
            share = history_share(),
            width = like.Width,
            height = like.Height,
            "the stand-in upscale mixes in the frame before (VICTORY_UPSCALE_HISTORY=0 for none)"
        );
        // (The first upscale after it was made has nothing to mix in.)
        *kept = Some(History { texture: texture.clone(), size, format: like.Format.0 });
        Some((texture, true))
    }

    fn pipeline(&self, device: &ID3D12Device, format: DXGI_FORMAT) -> Option<ID3D12PipelineState> {
        let mut made = self.pipelines.lock().unwrap_or_else(|e| e.into_inner());
        if let Some((_, p)) = made.iter().find(|(f, _)| *f == format.0) {
            return Some(p.clone());
        }
        let code = |b: &ID3DBlob| D3D12_SHADER_BYTECODE {
            // SAFETY: the blob lives as long as the upscaler.
            pShaderBytecode: unsafe { b.GetBufferPointer() },
            // SAFETY: as above.
            BytecodeLength: unsafe { b.GetBufferSize() },
        };
        let mut desc = D3D12_GRAPHICS_PIPELINE_STATE_DESC {
            // SAFETY: borrowed for the call; the upscaler holds the root.
            pRootSignature: unsafe { std::mem::transmute_copy(&self.root) },
            VS: code(&self.vs),
            PS: code(&self.ps),
            SampleMask: u32::MAX,
            RasterizerState: D3D12_RASTERIZER_DESC {
                FillMode: D3D12_FILL_MODE_SOLID,
                CullMode: D3D12_CULL_MODE_NONE,
                DepthClipEnable: false.into(),
                ..Default::default()
            },
            PrimitiveTopologyType: D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE,
            NumRenderTargets: 1,
            SampleDesc: DXGI_SAMPLE_DESC {
                Count: 1,
                Quality: 0,
            },
            ..Default::default()
        };
        desc.RTVFormats[0] = format;
        desc.BlendState.RenderTarget[0].RenderTargetWriteMask = D3D12_COLOR_WRITE_ENABLE_ALL.0 as u8;
        // SAFETY: a complete description with live shaders and root.
        match unsafe { device.CreateGraphicsPipelineState::<ID3D12PipelineState>(&desc) } {
            Ok(p) => {
                made.push((format.0, p.clone()));
                Some(p)
            }
            Err(e) => {
                tracing::warn!(error = %e, format = format.0, "the stand-in upscale pipeline refused");
                None
            }
        }
    }
}

fn upscaler() -> Option<&'static Upscaler> {
    static UPSCALER: std::sync::OnceLock<Option<Upscaler>> = std::sync::OnceLock::new();
    UPSCALER
        .get_or_init(|| {
            let host = crate::host::host()?;
            make(&host.device)
                .inspect_err(|e| tracing::warn!(error = %e, "no stand-in upscale"))
                .ok()
        })
        .as_ref()
}

/// Before [`UPSCALED_READER`] reads `target`: fills it with the last scene
/// scaled up, where no upscaler did. True if anything was recorded (the
/// list's state then needs setting again).
pub(crate) fn fill(list: &HostList, target: This) -> bool {
    let scene = LAST_SCENE.load(Ordering::Relaxed) as This;
    if !on() || scene.is_null() || target.is_null() || scene == target {
        return false;
    }
    if !crate::resource::stands_in(target) {
        return false;
    }
    LAST_TARGET.store(target as usize, Ordering::Relaxed);
    if WRITTEN.load(Ordering::Relaxed) {
        static ONCE: std::sync::Once = std::sync::Once::new();
        ONCE.call_once(|| tracing::info!("an upscaler wrote its output: the stand-in leaves it"));
        return false;
    }
    let (Some(host), Some(from), Some(to), Some(u)) = (
        crate::host::host(),
        crate::resource::host_resource(scene),
        crate::resource::host_resource(target),
        upscaler(),
    ) else {
        return false;
    };
    // SAFETY: live resources.
    let (s, d) = unsafe { (from.GetDesc(), to.GetDesc()) };
    if d.Width <= s.Width || d.Dimension != D3D12_RESOURCE_DIMENSION_TEXTURE2D || s.Dimension != d.Dimension {
        return false;
    }
    let Some(pso) = u.pipeline(&host.device, d.Format) else {
        return false;
    };
    {
        static ONCE: std::sync::Once = std::sync::Once::new();
        ONCE.call_once(|| {
            tracing::info!(
                from = format_args!("{}x{} f{}", s.Width, s.Height, s.Format.0),
                to = format_args!("{}x{} f{}", d.Width, d.Height, d.Format.0),
                "no upscaler ran: the scene is scaled up in its place (VICTORY_UPSCALE=0 for none)"
            );
        });
    }
    const PIXEL_SHADER_RESOURCE: u32 = 0x80;
    const RENDER_TARGET: u32 = 0x4;
    crate::list::ensure_state(list, scene, PIXEL_SHADER_RESOURCE);
    crate::list::ensure_state(list, target, RENDER_TARGET);
    crate::texture::host_written(target);
    // The scene's view names a typed format (a typeless host texture takes
    // the title's).
    let format = crate::resource::view_format(scene)
        .map_or(s.Format, |f| DXGI_FORMAT(crate::resource::pc_format(f) as i32));
    let share = history_share();
    let history = (share > 0.0).then(|| u.history(&host.device, &d)).flatten();
    let mix = match &history {
        Some((_, false)) => share,
        _ => 0.0,
    };
    let slot = (u.next.fetch_add(1, Ordering::Relaxed) % (SLOTS / 2)) * 2;
    let l = &list.list;
    // SAFETY: an open list; the view is made in a slot no list in flight
    // reads; live textures in the states the draw needs.
    unsafe {
        let cpu = D3D12_CPU_DESCRIPTOR_HANDLE {
            ptr: u.views.GetCPUDescriptorHandleForHeapStart().ptr + (slot * u.step) as usize,
        };
        let gpu = D3D12_GPU_DESCRIPTOR_HANDLE {
            ptr: u.views.GetGPUDescriptorHandleForHeapStart().ptr + u64::from(slot * u.step),
        };
        host.device.CreateShaderResourceView(
            from,
            Some(&D3D12_SHADER_RESOURCE_VIEW_DESC {
                Format: format,
                ViewDimension: D3D12_SRV_DIMENSION_TEXTURE2D,
                Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
                Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
                    Texture2D: D3D12_TEX2D_SRV {
                        MipLevels: 1,
                        ..Default::default()
                    },
                },
            }),
            cpu,
        );
        // What was made the frame before (a null view without it: the
        // shader then does not read it).
        host.device.CreateShaderResourceView(
            history.as_ref().map(|(h, _)| h),
            Some(&D3D12_SHADER_RESOURCE_VIEW_DESC {
                Format: d.Format,
                ViewDimension: D3D12_SRV_DIMENSION_TEXTURE2D,
                Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
                Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
                    Texture2D: D3D12_TEX2D_SRV {
                        MipLevels: 1,
                        ..Default::default()
                    },
                },
            }),
            D3D12_CPU_DESCRIPTOR_HANDLE { ptr: cpu.ptr + u.step as usize },
        );
        let rtv = host.rtv(Some(to), None);
        l.OMSetRenderTargets(1, Some(&rtv), false, None);
        l.SetDescriptorHeaps(&[Some(u.views.clone())]);
        l.SetGraphicsRootSignature(&u.root);
        l.SetPipelineState(&pso);
        l.SetGraphicsRootDescriptorTable(0, gpu);
        l.SetGraphicsRoot32BitConstant(1, mix.to_bits(), 0);
        l.IASetPrimitiveTopology(windows::Win32::Graphics::Direct3D::D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
        l.RSSetViewports(&[D3D12_VIEWPORT {
            Width: d.Width as f32,
            Height: d.Height as f32,
            MaxDepth: 1.0,
            ..Default::default()
        }]);
        l.RSSetScissorRects(&[windows::Win32::Foundation::RECT {
            left: 0,
            top: 0,
            right: d.Width as i32,
            bottom: d.Height as i32,
        }]);
        l.DrawInstanced(3, 1, 0, 0);
    }
    if let Some((h, _)) = &history {
        const COPY_SOURCE: u32 = 0x800;
        crate::list::ensure_state(list, target, COPY_SOURCE);
        let location = |r: &ID3D12Resource| D3D12_TEXTURE_COPY_LOCATION {
            // SAFETY: a borrowed pointer; the location never releases it.
            pResource: unsafe { std::mem::transmute_copy(r) },
            Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
            Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { SubresourceIndex: 0 },
        };
        let (read, copied) = (D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_COPY_DEST);
        // SAFETY: an open list; the history is readable between upscales
        // and the target is a copy source now; both live (the list keeps the
        // history).
        unsafe {
            l.ResourceBarrier(&[crate::host::transition(h, read, copied)]);
            l.CopyTextureRegion(&location(h), 0, 0, 0, &location(to), None);
            l.ResourceBarrier(&[crate::host::transition(h, copied, read)]);
        }
        list.keep(h);
    }
    list.keep(&u.views);
    list.keep(&pso);
    true
}

#[cfg(test)]
mod tests {
    use super::parse_share;

    #[test]
    fn the_share_of_the_frame_before() {
        assert_eq!(parse_share(""), 0.7);
        assert_eq!(parse_share("0"), 0.0);
        assert_eq!(parse_share(" 0.5 "), 0.5);
        assert_eq!(parse_share("2"), 0.95);
        assert_eq!(parse_share("-1"), 0.0);
        assert_eq!(parse_share("nan"), 0.7);
    }
}

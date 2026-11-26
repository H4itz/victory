//! A stand-in for the title's final composite, where it gives a flat picture.
//!
//! Grand Theft Auto VI turns its HDR scene into the picture in one pixel
//! shader (`PS_Composite`): exposure, bloom, a local tone curve from a
//! bilateral grid of the scene's luminance and baked 3D colour tables. On an
//! RTX 3050 (Series S, frame dump of run 204a979d) its output was one flat
//! grey over the whole screen while the scene, the bloom and the exposure it
//! reads all held the picture: something it reads (the grid comes from the
//! only wave64 program of the chain, the tables from compute programs that
//! write 3D textures) leaves nothing of the scene.
//!
//! With `VICTORY_TONEMAP=own`, that draw is replaced by ours: the scene
//! (the view of the target's size) plus some of the bloom (the view of half
//! its size), exposed by a histogram of the scene's luminance taken in the
//! vertex shader, ACES's fitted curve, sRGB. No colour grading, no local
//! tone curve, no lens effects of the composite; the passes after it (depth
//! of field, the interface, dithering) are the title's.
//! `VICTORY_TONEMAP=own:key,bloom,low,high` sets the mid grey the measured
//! luminance is brought to (0.18), the share of bloom (0.04), and the
//! shares of the darkest and brightest samples the measure leaves out
//! (0.5 and 0.05).
//! shortcut: a fixed curve in place of the title's grading.

use std::sync::OnceLock;
use std::sync::atomic::{AtomicU32, Ordering};
use windows::Win32::Graphics::Direct3D::Fxc::D3DCompile;
use windows::Win32::Graphics::Direct3D::ID3DBlob;
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::*;
use windows::core::{PCSTR, s};

use crate::host::HostList;
use crate::object::This;

/// The title's final composite, by the start of its name.
pub(crate) const PROGRAM: &[u8] = b"PS_Composite_";
/// Its pixel shader, by hash (Series 1.0.2.1), should the name not be found.
pub(crate) const KNOWN: &[&str] = &["7c649c422f6208e4"];

/// Knobs: key, bloom, low share, high share.
fn knobs() -> Option<[f32; 4]> {
    static KNOBS: OnceLock<Option<[f32; 4]>> = OnceLock::new();
    *KNOBS.get_or_init(|| {
        let set = std::env::var("VICTORY_TONEMAP").unwrap_or_default();
        let set = set.trim();
        let (mode, rest) = set.split_once(':').unwrap_or((set, ""));
        if mode != "own" && mode != "1" {
            return None;
        }
        let mut knobs = [0.18, 0.04, 0.5, 0.05];
        for (to, v) in knobs.iter_mut().zip(rest.split(',')) {
            if let Ok(v) = v.trim().parse::<f32>()
                && v.is_finite()
                && v >= 0.0
            {
                *to = v;
            }
        }
        knobs[2] = knobs[2].min(0.95);
        knobs[3] = knobs[3].min(0.95 - knobs[2]);
        tracing::info!(
            key = knobs[0],
            bloom = knobs[1],
            low = knobs[2],
            high = knobs[3],
            "the title's final composite is replaced by our tonemap (VICTORY_TONEMAP)"
        );
        Some(knobs)
    })
}

pub(crate) fn on() -> bool {
    knobs().is_some()
}

const SOURCE: &str = r#"
#define RS "RootConstants(num32BitConstants=8, b0), DescriptorTable(SRV(t0, numDescriptors=2)), StaticSampler(s0, filter=FILTER_MIN_MAG_MIP_LINEAR, addressU=TEXTURE_ADDRESS_CLAMP, addressV=TEXTURE_ADDRESS_CLAMP, addressW=TEXTURE_ADDRESS_CLAMP)"
cbuffer Knobs : register(b0) {
    float key;      // the mid grey the measured luminance is brought to
    float glow;     // the share of bloom
    float low;      // the share of the darkest samples left out
    float high;     // the share of the brightest samples left out
    float least_ev; // exposure limits, in stops
    float most_ev;
    float has_bloom;
    float unused;
};
Texture2D<float4> scene : register(t0);
Texture2D<float4> bloom : register(t1);
SamplerState smooth_ : register(s0);

struct V {
    float4 at : SV_Position;
    float2 uv : TEXCOORD0;
    nointerpolation float exposure : EXPOSURE;
};

#define GX 32
#define GY 18
#define BINS 64
#define LOW_LOG -10.0
#define HIGH_LOG 22.0

float luma(float3 c) { return dot(c, float3(0.2126, 0.7152, 0.0722)); }

float3 finite(float3 c) {
    return float3(c.r >= 0 && c.r < 65504 ? c.r : 0,
                  c.g >= 0 && c.g < 65504 ? c.g : 0,
                  c.b >= 0 && c.b < 65504 ? c.b : 0);
}

[RootSignature(RS)]
V vs(uint id : SV_VertexID) {
    V v;
    v.uv = float2((id << 1) & 2, id & 2);
    v.at = float4(v.uv * float2(2, -2) + float2(-1, 1), 0, 1);
    // A histogram of the scene's log luminance over a grid of samples,
    // those near the middle weighing more.
    float bins[BINS];
    [loop] for (uint b = 0; b < BINS; b++) bins[b] = 0;
    float total = 0;
    [loop] for (uint y = 0; y < GY; y++) {
        [loop] for (uint x = 0; x < GX; x++) {
            float2 at = (float2(x, y) + 0.5) / float2(GX, GY);
            float l = luma(finite(scene.SampleLevel(smooth_, at, 0).rgb));
            float2 off = (at - 0.5) * float2(1.0, 0.5625);
            float w = 1.5 - 2.0 * length(off);
            float slot = (log2(max(l, 1e-6)) - LOW_LOG) / (HIGH_LOG - LOW_LOG) * BINS;
            uint i = (uint)clamp(slot, 0, BINS - 1);
            bins[i] += w;
            total += w;
        }
    }
    // The mean log luminance of what lies between the darkest and the
    // brightest shares left out.
    float from = total * low, to = total * (1.0 - high), seen = 0, sum = 0, weight = 0;
    [loop] for (uint k = 0; k < BINS; k++) {
        float lo = max(seen, from), hi = min(seen + bins[k], to);
        if (hi > lo) {
            float centre = LOW_LOG + (k + 0.5) * (HIGH_LOG - LOW_LOG) / BINS;
            sum += (hi - lo) * centre;
            weight += hi - lo;
        }
        seen += bins[k];
    }
    float mean = weight > 0 ? sum / weight : 0;
    v.exposure = exp2(clamp(log2(max(key, 1e-6)) - mean, least_ev, most_ev));
    return v;
}

// ACES's fitted reference and output transforms (Stephen Hill).
static const float3x3 ACES_IN = {
    0.59719, 0.35458, 0.04823,
    0.07600, 0.90834, 0.01566,
    0.02840, 0.13383, 0.83777
};
static const float3x3 ACES_OUT = {
     1.60475, -0.53108, -0.07367,
    -0.10208,  1.10813, -0.00605,
    -0.00327, -0.07276,  1.07602
};

float3 fit(float3 v) {
    float3 a = v * (v + 0.0245786) - 0.000090537;
    float3 b = v * (0.983729 * v + 0.4329510) + 0.238081;
    return a / b;
}

float3 srgb(float3 c) {
    float3 lo = c * 12.92;
    float3 hi = 1.055 * pow(max(c, 1e-6), 1.0 / 2.4) - 0.055;
    return float3(c.r <= 0.0031308 ? lo.r : hi.r,
                  c.g <= 0.0031308 ? lo.g : hi.g,
                  c.b <= 0.0031308 ? lo.b : hi.b);
}

[RootSignature(RS)]
float4 ps(V v) : SV_Target {
    float3 c = finite(scene.SampleLevel(smooth_, v.uv, 0).rgb);
    if (has_bloom > 0) {
        c += glow * finite(bloom.SampleLevel(smooth_, v.uv, 0).rgb);
    }
    c *= v.exposure;
    c = saturate(mul(ACES_OUT, fit(mul(ACES_IN, c))));
    return float4(srgb(c), 1);
}
"#;

/// View pairs, used in turn: more than the draws of every list in flight.
const SLOTS: u32 = 64;

struct Tonemap {
    root: ID3D12RootSignature,
    vs: ID3DBlob,
    ps: ID3DBlob,
    pipelines: std::sync::Mutex<Vec<(i32, ID3D12PipelineState)>>,
    views: ID3D12DescriptorHeap,
    step: u32,
    next: AtomicU32,
}
// SAFETY: D3D12 objects are free-threaded; blobs are only read.
unsafe impl Send for Tonemap {}
// SAFETY: as above.
unsafe impl Sync for Tonemap {}

fn compile(entry: PCSTR, target: PCSTR) -> windows::core::Result<ID3DBlob> {
    let mut code: Option<ID3DBlob> = None;
    let mut errors: Option<ID3DBlob> = None;
    // SAFETY: the source and names live through the call; the out pointers
    // are ours.
    let made = unsafe {
        D3DCompile(
            SOURCE.as_ptr().cast(),
            SOURCE.len(),
            s!("tonemap"),
            None,
            None,
            entry,
            target,
            0,
            0,
            &mut code,
            Some(&mut errors),
        )
    };
    if let Err(e) = made {
        let text = errors
            .as_ref()
            .map(|b| String::from_utf8_lossy(bytes(b)).into_owned());
        tracing::warn!(error = %e, text = text.unwrap_or_default(), "the stand-in tonemap does not compile");
        return Err(e);
    }
    code.ok_or_else(windows::core::Error::empty)
}

fn bytes(b: &ID3DBlob) -> &[u8] {
    // SAFETY: the blob's own bytes, alive as long as it.
    unsafe { std::slice::from_raw_parts(b.GetBufferPointer().cast::<u8>(), b.GetBufferSize()) }
}

fn make(device: &ID3D12Device) -> windows::core::Result<Tonemap> {
    let (vs, ps) = (
        compile(s!("vs"), s!("vs_5_1"))?,
        compile(s!("ps"), s!("ps_5_1"))?,
    );
    // SAFETY: plain object creation from the compiler's blobs.
    unsafe {
        let root: ID3D12RootSignature = device.CreateRootSignature(0, bytes(&vs))?;
        let views = device.CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
            Type: D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
            NumDescriptors: SLOTS * 2,
            Flags: D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE,
            NodeMask: 0,
        })?;
        Ok(Tonemap {
            root,
            vs,
            ps,
            pipelines: std::sync::Mutex::new(Vec::new()),
            views,
            step: device.GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV),
            next: AtomicU32::new(0),
        })
    }
}

impl Tonemap {
    fn pipeline(&self, device: &ID3D12Device, format: DXGI_FORMAT) -> Option<ID3D12PipelineState> {
        let mut made = self.pipelines.lock().unwrap_or_else(|e| e.into_inner());
        if let Some((_, p)) = made.iter().find(|(f, _)| *f == format.0) {
            return Some(p.clone());
        }
        let code = |b: &ID3DBlob| D3D12_SHADER_BYTECODE {
            // SAFETY: the blob lives as long as the tonemap.
            pShaderBytecode: unsafe { b.GetBufferPointer() },
            // SAFETY: as above.
            BytecodeLength: unsafe { b.GetBufferSize() },
        };
        let mut desc = D3D12_GRAPHICS_PIPELINE_STATE_DESC {
            // SAFETY: borrowed for the call; the tonemap holds the root.
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
        desc.BlendState.RenderTarget[0].RenderTargetWriteMask =
            D3D12_COLOR_WRITE_ENABLE_ALL.0 as u8;
        // SAFETY: a complete description with live shaders and root.
        match unsafe { device.CreateGraphicsPipelineState::<ID3D12PipelineState>(&desc) } {
            Ok(p) => {
                made.push((format.0, p.clone()));
                Some(p)
            }
            Err(e) => {
                tracing::warn!(error = %e, format = format.0, "the stand-in tonemap pipeline refused");
                None
            }
        }
    }
}

fn tonemap() -> Option<&'static Tonemap> {
    static TONEMAP: OnceLock<Option<Tonemap>> = OnceLock::new();
    TONEMAP
        .get_or_init(|| {
            let host = crate::host::host()?;
            make(&host.device)
                .inspect_err(|e| tracing::warn!(error = %e, "no stand-in tonemap"))
                .ok()
        })
        .as_ref()
}

/// HDR colour formats a scene or its bloom is held in on the host.
const HDR: [i32; 4] = [
    DXGI_FORMAT_R32G32B32A32_FLOAT.0,
    DXGI_FORMAT_R16G16B16A16_FLOAT.0,
    DXGI_FORMAT_R11G11B10_FLOAT.0,
    DXGI_FORMAT_R9G9B9E5_SHAREDEXP.0,
];

/// The format to view `r` in: the host's for a stand-in, the title's
/// where the host texture is typeless.
fn view_format(r: This, d: &D3D12_RESOURCE_DESC) -> DXGI_FORMAT {
    if crate::resource::stands_in(r) {
        return d.Format;
    }
    crate::resource::view_format(r).map_or(d.Format, |f| {
        DXGI_FORMAT(crate::resource::pc_format(f) as i32)
    })
}

/// A plain 2D HDR texture: its description and the format to view it in.
fn hdr(r: This) -> Option<(&'static ID3D12Resource, D3D12_RESOURCE_DESC, DXGI_FORMAT)> {
    let texture = crate::resource::host_resource(r)?;
    // SAFETY: a live resource.
    let d = unsafe { texture.GetDesc() };
    if d.Dimension != D3D12_RESOURCE_DIMENSION_TEXTURE2D
        || d.DepthOrArraySize != 1
        || d.SampleDesc.Count != 1
    {
        return None;
    }
    let format = view_format(r, &d);
    HDR.contains(&format.0).then_some((texture, d, format))
}

/// In place of the title's composite drawing into `target`: the scene among
/// the textures it reads (`reads`), tonemapped. True if drawn; the list's
/// state then needs setting again.
pub(crate) fn draw(list: &HostList, target: This, reads: &[This]) -> bool {
    let Some(knobs) = knobs() else {
        return false;
    };
    let (Some(host), Some(to), Some(t)) = (
        crate::host::host(),
        crate::resource::host_resource(target),
        tonemap(),
    ) else {
        return false;
    };
    // SAFETY: a live resource.
    let d = unsafe { to.GetDesc() };
    let size = |e: &D3D12_RESOURCE_DESC| (e.Width, e.Height);
    let scene = reads.iter().filter(|&&r| r != target).find_map(|&r| {
        hdr(r)
            .filter(|(_, e, _)| size(e) == size(&d))
            .map(|v| (r, v))
    });
    let Some((scene, (from, _, scene_format))) = scene else {
        static ONCE: std::sync::Once = std::sync::Once::new();
        ONCE.call_once(|| {
            tracing::warn!(
                width = d.Width,
                height = d.Height,
                "the composite reads no scene of its target's size: its own draw stays"
            );
        });
        return false;
    };
    let half = (d.Width.div_ceil(2), d.Height.div_ceil(2));
    let bloom = reads
        .iter()
        .filter(|&&r| r != target && r != scene)
        .find_map(|&r| {
            hdr(r)
                .filter(|(_, e, _)| {
                    e.Width.abs_diff(half.0) <= 1
                        && u64::from(e.Height).abs_diff(u64::from(half.1)) <= 1
                })
                .map(|v| (r, v))
        });
    let Some(pso) = t.pipeline(&host.device, d.Format) else {
        return false;
    };
    {
        static ONCE: std::sync::Once = std::sync::Once::new();
        ONCE.call_once(|| {
            tracing::info!(
                to = format_args!("{}x{} f{}", d.Width, d.Height, d.Format.0),
                scene = scene_format.0,
                bloom = bloom.is_some(),
                "our tonemap draws in place of the title's composite"
            );
        });
    }
    const PIXEL_SHADER_RESOURCE: u32 = 0x80;
    const RENDER_TARGET: u32 = 0x4;
    crate::list::ensure_state(list, scene, PIXEL_SHADER_RESOURCE);
    if let Some((b, _)) = bloom {
        crate::list::ensure_state(list, b, PIXEL_SHADER_RESOURCE);
    }
    crate::list::ensure_state(list, target, RENDER_TARGET);
    crate::texture::host_written(target);
    let slot = t.next.fetch_add(1, Ordering::Relaxed) % SLOTS;
    let constants: [f32; 8] = [
        knobs[0],
        knobs[1],
        knobs[2],
        knobs[3],
        -16.0,
        4.0,
        if bloom.is_some() { 1.0 } else { 0.0 },
        0.0,
    ];
    let l = &list.list;
    // SAFETY: an open list; the views are made in slots no list in flight
    // reads; live textures in the states the draw needs.
    unsafe {
        let cpu = |i: u32| D3D12_CPU_DESCRIPTOR_HANDLE {
            ptr: t.views.GetCPUDescriptorHandleForHeapStart().ptr
                + ((slot * 2 + i) * t.step) as usize,
        };
        let gpu = D3D12_GPU_DESCRIPTOR_HANDLE {
            ptr: t.views.GetGPUDescriptorHandleForHeapStart().ptr + u64::from(slot * 2 * t.step),
        };
        let srv = |format: DXGI_FORMAT| D3D12_SHADER_RESOURCE_VIEW_DESC {
            Format: format,
            ViewDimension: D3D12_SRV_DIMENSION_TEXTURE2D,
            Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
            Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
                Texture2D: D3D12_TEX2D_SRV {
                    MipLevels: 1,
                    ..Default::default()
                },
            },
        };
        host.device
            .CreateShaderResourceView(from, Some(&srv(scene_format)), cpu(0));
        match bloom {
            Some((_, (texture, _, format))) => {
                host.device
                    .CreateShaderResourceView(texture, Some(&srv(format)), cpu(1));
            }
            None => {
                host.device.CreateShaderResourceView(
                    None,
                    Some(&srv(DXGI_FORMAT_R16G16B16A16_FLOAT)),
                    cpu(1),
                );
            }
        }
        let rtv = host.rtv(to, None);
        l.OMSetRenderTargets(1, Some(&rtv), false, None);
        l.SetDescriptorHeaps(&[Some(t.views.clone())]);
        l.SetGraphicsRootSignature(&t.root);
        l.SetPipelineState(&pso);
        l.SetGraphicsRoot32BitConstants(0, constants.len() as u32, constants.as_ptr().cast(), 0);
        l.SetGraphicsRootDescriptorTable(1, gpu);
        l.IASetPrimitiveTopology(
            windows::Win32::Graphics::Direct3D::D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST,
        );
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
    list.keep(&t.views);
    list.keep(&pso);
    true
}

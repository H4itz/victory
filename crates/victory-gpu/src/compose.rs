//! UI layer composition (`VICTORY_COMPOSE_UI`, see [`crate::uilayer`]):
//! layers drawn one over another into the swap chain's back buffer, each
//! stretched to it, the first opaque and the rest blended by their alpha
//! (premultiplied, or straight on request). A full-screen triangle samples
//! one layer per draw; the root signature, pipeline and descriptor heaps
//! are ours, made once, on our own present list.

use windows::Win32::Graphics::Direct3D::Fxc::D3DCompile;
use windows::Win32::Graphics::Direct3D::ID3DBlob;
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::*;
use windows::core::PCSTR;

use crate::host::transition;

const SOURCE: &str = r#"
#define RS "RootConstants(num32BitConstants=2, b0), DescriptorTable(SRV(t0)), StaticSampler(s0, filter=FILTER_MIN_MAG_MIP_LINEAR, addressU=TEXTURE_ADDRESS_CLAMP, addressV=TEXTURE_ADDRESS_CLAMP, addressW=TEXTURE_ADDRESS_CLAMP)"
cbuffer Layer : register(b0) { uint opaque; uint straight; };
Texture2D<float4> layer : register(t0);
SamplerState linear_clamp : register(s0);
struct V { float4 position : SV_Position; float2 uv : TEXCOORD0; };
[RootSignature(RS)]
V vs(uint id : SV_VertexID) {
    V o;
    o.uv = float2((id << 1) & 2, id & 2);
    o.position = float4(o.uv * float2(2, -2) + float2(-1, 1), 0, 1);
    return o;
}
[RootSignature(RS)]
float4 ps(V i) : SV_Target {
    float4 c = saturate(layer.Sample(linear_clamp, i.uv));
    if (opaque != 0) { c.a = 1; }
    else if (straight != 0) { c.rgb *= c.a; }
    return c;
}
"#;

/// Layers drawn at most per present.
const LAYERS: u32 = 2;

pub(crate) struct Compose {
    root: ID3D12RootSignature,
    pso: ID3D12PipelineState,
    srvs: ID3D12DescriptorHeap,
    rtv: ID3D12DescriptorHeap,
    srv_step: u32,
}

// SAFETY: D3D12 objects are free-threaded; presents are serialised by the
// presenter's lock.
unsafe impl Send for Compose {}
// SAFETY: as above.
unsafe impl Sync for Compose {}

/// The back buffer format composition renders to.
pub(crate) const FORMAT: DXGI_FORMAT = DXGI_FORMAT_R8G8B8A8_UNORM;

pub(crate) fn get(device: &ID3D12Device) -> Option<&'static Compose> {
    static COMPOSE: std::sync::OnceLock<Option<Compose>> = std::sync::OnceLock::new();
    COMPOSE
        .get_or_init(|| match make(device) {
            Ok(c) => Some(c),
            Err(e) => {
                tracing::warn!(error = %e, "no UI composition: showing the layer alone");
                None
            }
        })
        .as_ref()
}

fn compile(entry: PCSTR, target: PCSTR) -> windows::core::Result<ID3DBlob> {
    let mut code: Option<ID3DBlob> = None;
    let mut errors: Option<ID3DBlob> = None;
    // SAFETY: the source and names live through the call; out pointers ours.
    let compiled = unsafe {
        D3DCompile(
            SOURCE.as_ptr().cast(),
            SOURCE.len(),
            windows::core::s!("compose"),
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
    if let Err(e) = compiled {
        if let Some(errors) = errors {
            // SAFETY: the blob's own bytes.
            let text = unsafe {
                std::slice::from_raw_parts(errors.GetBufferPointer().cast::<u8>(), errors.GetBufferSize())
            };
            tracing::warn!("UI composition shader: {}", String::from_utf8_lossy(text));
        }
        return Err(e);
    }
    code.ok_or_else(windows::core::Error::empty)
}

fn bytes(blob: &ID3DBlob) -> &[u8] {
    // SAFETY: the blob's own bytes, alive as long as the blob.
    unsafe { std::slice::from_raw_parts(blob.GetBufferPointer().cast::<u8>(), blob.GetBufferSize()) }
}

fn make(device: &ID3D12Device) -> windows::core::Result<Compose> {
    let vs = compile(windows::core::s!("vs"), windows::core::s!("vs_5_1"))?;
    let ps = compile(windows::core::s!("ps"), windows::core::s!("ps_5_1"))?;
    // SAFETY: plain object creation from our own compiled blobs.
    unsafe {
        let root: ID3D12RootSignature = device.CreateRootSignature(0, bytes(&vs))?;
        let blend = D3D12_RENDER_TARGET_BLEND_DESC {
            BlendEnable: true.into(),
            LogicOpEnable: false.into(),
            SrcBlend: D3D12_BLEND_ONE,
            DestBlend: D3D12_BLEND_INV_SRC_ALPHA,
            BlendOp: D3D12_BLEND_OP_ADD,
            SrcBlendAlpha: D3D12_BLEND_ONE,
            DestBlendAlpha: D3D12_BLEND_INV_SRC_ALPHA,
            BlendOpAlpha: D3D12_BLEND_OP_ADD,
            LogicOp: D3D12_LOGIC_OP_NOOP,
            RenderTargetWriteMask: D3D12_COLOR_WRITE_ENABLE_ALL.0 as u8,
        };
        let mut formats = [DXGI_FORMAT_UNKNOWN; 8];
        formats[0] = FORMAT;
        let desc = D3D12_GRAPHICS_PIPELINE_STATE_DESC {
            pRootSignature: std::mem::transmute_copy(&root),
            VS: D3D12_SHADER_BYTECODE {
                pShaderBytecode: bytes(&vs).as_ptr().cast(),
                BytecodeLength: bytes(&vs).len(),
            },
            PS: D3D12_SHADER_BYTECODE {
                pShaderBytecode: bytes(&ps).as_ptr().cast(),
                BytecodeLength: bytes(&ps).len(),
            },
            BlendState: D3D12_BLEND_DESC {
                AlphaToCoverageEnable: false.into(),
                IndependentBlendEnable: false.into(),
                RenderTarget: [blend; 8],
            },
            SampleMask: u32::MAX,
            RasterizerState: D3D12_RASTERIZER_DESC {
                FillMode: D3D12_FILL_MODE_SOLID,
                CullMode: D3D12_CULL_MODE_NONE,
                DepthClipEnable: true.into(),
                ..Default::default()
            },
            PrimitiveTopologyType: D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE,
            NumRenderTargets: 1,
            RTVFormats: formats,
            SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
            ..Default::default()
        };
        let pso: ID3D12PipelineState = device.CreateGraphicsPipelineState(&desc)?;
        let srvs: ID3D12DescriptorHeap = device.CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
            Type: D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
            NumDescriptors: LAYERS,
            Flags: D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE,
            NodeMask: 0,
        })?;
        let rtv: ID3D12DescriptorHeap = device.CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
            Type: D3D12_DESCRIPTOR_HEAP_TYPE_RTV,
            NumDescriptors: 1,
            Flags: D3D12_DESCRIPTOR_HEAP_FLAG_NONE,
            NodeMask: 0,
        })?;
        let srv_step = device.GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
        tracing::info!("UI composition ready");
        Ok(Compose { root, pso, srvs, rtv, srv_step })
    }
}

/// A typed format to sample a layer through (typeless layers: their
/// family's first typed format).
fn view_format(format: DXGI_FORMAT) -> Option<DXGI_FORMAT> {
    Some(match format.0 {
        9 | 23 | 27 => DXGI_FORMAT(format.0 + 1),
        90 => DXGI_FORMAT_B8G8R8A8_UNORM,
        10..=14 | 24 | 25 | 28..=32 | 87 | 91 => format,
        _ => return None,
    })
}

/// A layer to draw: its texture and the state it is in (it is left there).
pub(crate) struct Layer<'a> {
    pub(crate) texture: &'a ID3D12Resource,
    pub(crate) state: D3D12_RESOURCE_STATES,
}

impl Compose {
    /// Records, in `list`, `layers` drawn over one another into `back` (in
    /// the present state, left there): the first opaque, the rest blended.
    /// Layers it cannot sample are skipped. False if none was drawn.
    pub(crate) fn record(
        &self,
        device: &ID3D12Device,
        list: &ID3D12GraphicsCommandList,
        back: &ID3D12Resource,
        (width, height): (u32, u32),
        layers: &[Layer<'_>],
        straight: bool,
    ) -> bool {
        let usable: Vec<(&Layer<'_>, DXGI_FORMAT)> = layers
            .iter()
            .take(LAYERS as usize)
            .filter_map(|l| {
                // SAFETY: a live resource.
                let d = unsafe { l.texture.GetDesc() };
                let plain = d.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE2D
                    && d.SampleDesc.Count == 1
                    && d.DepthOrArraySize == 1
                    && d.Flags.0 & D3D12_RESOURCE_FLAG_DENY_SHADER_RESOURCE.0 == 0;
                plain.then(|| view_format(d.Format)).flatten().map(|f| (l, f))
            })
            .collect();
        if usable.is_empty() {
            return false;
        }
        let read = D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE;
        // SAFETY: an open list of ours; live resources; our heaps, made
        // for this many views.
        unsafe {
            let srv_cpu = self.srvs.GetCPUDescriptorHandleForHeapStart();
            let srv_gpu = self.srvs.GetGPUDescriptorHandleForHeapStart();
            for (i, (layer, format)) in usable.iter().enumerate() {
                let view = D3D12_SHADER_RESOURCE_VIEW_DESC {
                    Format: *format,
                    ViewDimension: D3D12_SRV_DIMENSION_TEXTURE2D,
                    Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
                    Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
                        Texture2D: D3D12_TEX2D_SRV {
                            MostDetailedMip: 0,
                            MipLevels: 1,
                            PlaneSlice: 0,
                            ResourceMinLODClamp: 0.0,
                        },
                    },
                };
                let at = D3D12_CPU_DESCRIPTOR_HANDLE {
                    ptr: srv_cpu.ptr + i * self.srv_step as usize,
                };
                device.CreateShaderResourceView(layer.texture, Some(&view), at);
            }
            let rtv = self.rtv.GetCPUDescriptorHandleForHeapStart();
            device.CreateRenderTargetView(back, None, rtv);
            let mut before: Vec<D3D12_RESOURCE_BARRIER> = usable
                .iter()
                .filter(|(l, _)| l.state != read)
                .map(|(l, _)| transition(l.texture, l.state, read))
                .collect();
            before.push(transition(
                back,
                D3D12_RESOURCE_STATE_PRESENT,
                D3D12_RESOURCE_STATE_RENDER_TARGET,
            ));
            list.ResourceBarrier(&before);
            list.OMSetRenderTargets(1, Some(&rtv), false, None);
            list.ClearRenderTargetView(rtv, &[0.0, 0.0, 0.0, 1.0], None);
            list.RSSetViewports(&[D3D12_VIEWPORT {
                TopLeftX: 0.0,
                TopLeftY: 0.0,
                Width: width as f32,
                Height: height as f32,
                MinDepth: 0.0,
                MaxDepth: 1.0,
            }]);
            list.RSSetScissorRects(&[windows::Win32::Foundation::RECT {
                left: 0,
                top: 0,
                right: width as i32,
                bottom: height as i32,
            }]);
            list.SetDescriptorHeaps(&[Some(self.srvs.clone())]);
            list.SetGraphicsRootSignature(&self.root);
            list.SetPipelineState(&self.pso);
            list.IASetPrimitiveTopology(windows::Win32::Graphics::Direct3D::D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
            for i in 0..usable.len() {
                let constants = [u32::from(i == 0), u32::from(straight)];
                list.SetGraphicsRoot32BitConstants(0, 2, constants.as_ptr().cast(), 0);
                list.SetGraphicsRootDescriptorTable(
                    1,
                    D3D12_GPU_DESCRIPTOR_HANDLE {
                        ptr: srv_gpu.ptr + (i as u64) * u64::from(self.srv_step),
                    },
                );
                list.DrawInstanced(3, 1, 0, 0);
            }
            let mut after: Vec<D3D12_RESOURCE_BARRIER> = usable
                .iter()
                .filter(|(l, _)| l.state != read)
                .map(|(l, _)| transition(l.texture, read, l.state))
                .collect();
            after.push(transition(
                back,
                D3D12_RESOURCE_STATE_RENDER_TARGET,
                D3D12_RESOURCE_STATE_PRESENT,
            ));
            list.ResourceBarrier(&after);
        }
        true
    }
}

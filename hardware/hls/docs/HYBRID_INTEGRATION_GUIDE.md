# Hybrid HLS-Verilog 통합 가이드

## 아키텍처 개요

```
┌────────────────────────────────────────────────────────────────────────┐
│                         PYNQ-Z2 (Zynq-7020)                            │
│  ┌──────────────────┐       ┌────────────────────────────────────────┐ │
│  │   PS (ARM)       │       │              PL (FPGA)                 │ │
│  │                  │       │  ┌──────────────────────────────────┐  │ │
│  │  Python/C App    │       │  │     axi_hls_wrapper (HLS IP)     │  │ │
│  │                  │ AXI   │  │  ┌────────────┐  ┌────────────┐  │  │ │
│  │  ┌────────────┐  │◄─────►│  │  │ AXI4-Lite  │  │ AXI-Stream │  │  │ │
│  │  │ PYNQ       │  │       │  │  │ Registers  │  │ Interface  │  │  │ │
│  │  │ Overlay    │  │       │  │  └─────┬──────┘  └──────┬─────┘  │  │ │
│  │  └────────────┘  │       │  │        │                │        │  │ │
│  └──────────────────┘       │  │        │    Wire I/F    │        │  │ │
│                             │  └────────┼────────────────┼────────┘  │ │
│                             │           │                │           │ │
│                             │           ▼                ▼           │ │
│                             │  ┌────────────────────────────────────┐│ │
│                             │  │        Verilog SNN Core            ││ │
│                             │  │  ┌──────────┐  ┌──────────────┐    ││ │
│                             │  │  │ synapse  │  │ spike_router │    ││ │
│                             │  │  │ _array   │  │              │    ││ │
│                             │  │  └──────────┘  └──────────────┘    ││ │
│                             │  │  ┌──────────┐  ┌──────────────┐    ││ │
│                             │  │  │ lif_     │  │  snn_conv1d  │    ││ │
│                             │  │  │ neuron   │  │              │    ││ │
│                             │  │  └──────────┘  └──────────────┘    ││ │
│                             │  └────────────────────────────────────┘│ │
│                             └────────────────────────────────────────┘ │
└────────────────────────────────────────────────────────────────────────┘
```

## 파일 구조

```
hardware/
├── hls/
│   ├── include/
│   │   ├── axi_hls_wrapper.h       # HLS wrapper 헤더
│   │   ├── axi_interfaces.h        # AXI 타입 정의
│   │   └── snn_types.h             # SNN 데이터 타입
│   ├── src/
│   │   └── axi_hls_wrapper.cpp     # HLS wrapper 구현
│   ├── test/
│   │   └── tb_axi_hls_wrapper.cpp  # C++ 테스트벤치
│   └── scripts/
│       └── run_hls_wrapper.tcl     # Vitis HLS 빌드 스크립트
├── hdl/
│   └── rtl/
│       ├── top/
│       │   └── snn_top_hybrid.v    # Hybrid 통합 Top 모듈
│       ├── neurons/
│       ├── synapses/
│       └── router/
└── vivado/
    └── block_design.tcl            # Block Design 생성 스크립트
```

## 통합 단계

### 1. HLS IP 합성

```bash
cd hardware/hls/scripts
vitis_hls -f run_hls_wrapper.tcl
```

생성 결과:
- `hardware/hls/proj_axi_wrapper/solution1/impl/ip/` - IP 패키지

### 2. Vivado IP Repository 등록

Vivado에서:
1. Settings → IP → Repository
2. `hardware/hls/proj_axi_wrapper/solution1/impl/ip/` 추가

### 3. Block Design 생성

```tcl
# Block Design에서:
# 1. ZYNQ PS 추가
# 2. axi_hls_wrapper IP 추가
# 3. snn_core RTL 모듈 추가 (Add Module)
# 4. 연결:
#    - PS M_AXI_GP0 → HLS wrapper s_axi
#    - PS M_AXIS_S2MM → HLS wrapper s_axis_spikes
#    - HLS wrapper m_axis_spikes → PS S_AXIS_MM2S
#    - HLS wrapper wire 출력 → Verilog SNN core
```

### 4. Verilog 연결 인터페이스

HLS wrapper와 Verilog 모듈 간 연결:

| HLS Port | Direction | Verilog Port | Description |
|----------|-----------|--------------|-------------|
| spike_in_valid | out | spike_in_valid | 입력 스파이크 유효 |
| spike_in_neuron_id | out | spike_in_axon_id | 뉴런/액손 ID |
| spike_in_weight | out | spike_in_weight | 가중치 |
| spike_in_ready | in | spike_in_ready | Ready 신호 |
| spike_out_valid | in | spike_out_valid | 출력 스파이크 유효 |
| spike_out_neuron_id | in | spike_out_neuron_id | 출력 뉴런 ID |
| spike_out_weight | in | spike_out_weight | 출력 가중치 |
| spike_out_ready | out | spike_out_ready | Ready 신호 |
| snn_enable | out | enable | SNN 활성화 |
| snn_reset | out | rst_n (inverted) | 리셋 |
| leak_rate_out | out | leak_rate | Leak rate |
| threshold_out | out | threshold | Threshold |

## 레지스터 맵

| Address | Name | R/W | Description |
|---------|------|-----|-------------|
| 0x00 | CTRL | R/W | Control register |
| 0x04 | STATUS | R | Status register |
| 0x08 | CONFIG | R/W | Configuration |
| 0x0C | SPIKE_COUNT | R | Spike counter |
| 0x10 | LEAK_RATE | R/W | Neuron leak rate |
| 0x14 | THRESHOLD | R/W | Neuron threshold |
| 0x18 | REFRACTORY | R/W | Refractory period |
| 0x1C | VERSION | R | Version ID |

### Control Register (0x00)

| Bit | Name | Description |
|-----|------|-------------|
| 0 | ENABLE | SNN enable |
| 1 | RESET | Soft reset |
| 2 | CLEAR | Clear counters |
| 3 | LEARNING | Enable learning |
| 4 | IRQ_EN | Interrupt enable |

## Python 사용 예제

```python
from pynq import Overlay, allocate
import numpy as np

# Load overlay
ol = Overlay("snn_accelerator.bit")

# Access HLS IP
snn = ol.axi_hls_wrapper_0

# Configure
snn.write(0x10, 10)    # leak_rate
snn.write(0x14, 1000)  # threshold
snn.write(0x18, 20)    # refractory

# Enable
snn.write(0x00, 0x01)

# Send spikes via DMA
dma = ol.axi_dma_0
input_buffer = allocate(shape=(100,), dtype=np.uint32)

# Pack spike: [31:24]=0, [23:16]=weight, [15:8]=0, [7:0]=neuron_id
for i in range(100):
    input_buffer[i] = (100 << 16) | i  # weight=100, neuron_id=i

dma.sendchannel.transfer(input_buffer)
dma.sendchannel.wait()

# Read status
status = snn.read(0x04)
spike_count = snn.read(0x0C)
print(f"Status: {status:#x}, Spike count: {spike_count}")
```

## 기존 Verilog 코드와의 호환성

### 변경 필요 사항

1. **axi_wrapper.v** - 제거 (HLS로 대체)
2. **axi_lite_regs_v1_0.v** - 제거 (HLS로 대체)
3. **snn_accelerator_top.v** - 수정 (새 Top 모듈로 대체)

### 유지되는 모듈

- `synapse_array.v` - 그대로 유지
- `spike_router.v` - 그대로 유지
- `lif_neuron.v` - 그대로 유지
- `snn_conv1d.v` - 그대로 유지
- `snn_maxpool1d.v` - 그대로 유지

## 빌드 Flow

```
1. HLS 합성:   vitis_hls -f run_hls_wrapper.tcl
2. IP 패키징:   (HLS에서 자동 생성)
3. Vivado:     vivado -mode batch -source build_hybrid.tcl
4. Bitstream:  생성된 .bit 파일
5. PYNQ:       .bit + .hwh 파일 업로드
```

## 장점

1. **AXI 프로토콜 자동화**: HLS가 복잡한 AXI 핸드셰이크 처리
2. **유연한 버퍼링**: HLS에서 FIFO 크기 조절 용이
3. **빠른 검증**: C++ 시뮬레이션으로 빠른 기능 검증
4. **기존 코드 재사용**: Verilog SNN 코어 모듈 그대로 사용
5. **확장성**: 새 기능 추가 시 HLS로 빠르게 프로토타이핑

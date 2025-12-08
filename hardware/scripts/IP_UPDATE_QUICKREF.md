# IP replace & Update Address Map Quick Reference

## Step 1: Check Register Offset

HLS가 생성한 주소 (상대 오프셋):

```
0x00  AP_CTRL         (start/done/idle/ready)
0x20  mode_reg        (bits[1:0]=mode, bit[8]=encoder_en)
0x28  time_steps_reg
0x30  learning_params[0..4]  (5 words)
0x48  encoder_config[0..3]   (4 words)
0x5C  status_reg      (read-only)
0x9C  reward_signal
```

Python RegisterMap은 이미 동기화됨: `software/python/snn_fpga_accelerator/xrt_backend.py`

---

## 2단계: Vivado CLI로 IP 업데이트

```bash
# 1. 검증
cd hardware/scripts
./verify_hls_ip.sh

# 2. IP 교체 (자동)
vivado -mode batch -source update_hls_ip.tcl

# 3. 콘솔에서 베이스 주소 확인 (예: 0x43C00000)
```

---

## 3단계: Python에서 베이스 주소 사용

```python
# PYNQ
from pynq import MMIO
mmio = MMIO(0x43C00000, 0x1000)  # Vivado Address Editor에서 확인한 베이스
mmio.write(0x20, 0x101)  # mode_reg: INFER + encoder

# XRT
from snn_fpga_accelerator.xrt_backend import XRTBackend
backend = XRTBackend("design.xclbin")
backend.set_mode(mode=1, encoder_enable=True)  # 자동으로 0x20에 write
```

---

## 4단계: 비트스트림 생성 (선택)

```tcl
# Vivado Tcl Console
open_project hardware/vivado/*.xpr
launch_runs impl_1 -to_step write_bitstream -jobs 8
wait_on_run impl_1
write_hw_platform -fixed -include_bit -file design.xsa
```

---

## 트러블슈팅

### Q: IP 업그레이드 후 "Missing port" 에러
**A:** `s_axis_data` 같은 새 포트는 tie-off 또는 DMA 연결 필요:
```tcl
# Tie-off 예시
create_bd_cell -type ip -vlnv xilinx.com:ip:xlconstant:1.1 tvalid_0
set_property CONFIG.CONST_VAL {0} [get_bd_cells tvalid_0]
connect_bd_net [get_bd_pins tvalid_0/dout] [get_bd_pins snn_top_hls_0/s_axis_data_TVALID]
```

### Q: Python write 무반응
**A:** `devmem`으로 물리 주소 직접 테스트:
```bash
devmem 0x43C00020 32  # mode_reg 읽기
devmem 0x43C00020 32 0x101  # mode_reg 쓰기
```

---

## 전체 문서

상세 설명 및 예제: [docs/HLS_IP_INTEGRATION_GUIDE.md](../docs/HLS_IP_INTEGRATION_GUIDE.md)

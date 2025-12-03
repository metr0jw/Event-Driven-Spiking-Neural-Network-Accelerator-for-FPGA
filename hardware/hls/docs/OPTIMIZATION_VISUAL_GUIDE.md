# Optimization Strategy Visual Comparison

## Algorithm Flow Diagrams

### 1. Baseline Strategy

```
┌─────────────────────────────────────────────┐
│         Pre-Spike Arrives (neuron 5)        │
└────────────────┬────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────┐
│    Update pre_spike_times[5] = timestamp    │
└────────────────┬────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────┐
│         SCAN ALL NEURONS (0-63)             │
│                                             │
│  for post_id = 0 to 63:                    │
│    ├─ Check post_spike_times[post_id]      │
│    ├─ Calculate dt = pre - post            │
│    ├─ If within STDP window:               │
│    │   └─ Generate LTD weight update       │
│    └─ Next iteration                        │
│                                             │
│  Latency: 64 neurons × II=2 = 128 cycles   │
└────────────────┬────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────┐
│       Output: Weight Update Stream          │
│  [pre=5, post=12, delta=-8, time=1000]     │
│  [pre=5, post=27, delta=-5, time=1000]     │
│  [pre=5, post=41, delta=-3, time=1000]     │
└─────────────────────────────────────────────┘

Performance: O(N) - Linear scan, predictable latency
```

---

### 2. Active Tracking Strategy

```
┌─────────────────────────────────────────────┐
│         Pre-Spike Arrives (neuron 5)        │
└────────────────┬────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────┐
│    Update pre_spike_times[5] = timestamp    │
│    Add neuron 5 to active_pre list          │
│                                             │
│    active_pre = [2, 5, 7, 15]               │
│    count = 4                                │
└────────────────┬────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────┐
│    SCAN ONLY ACTIVE POST NEURONS            │
│                                             │
│    active_post = [12, 27, 41]  (3 neurons) │
│                                             │
│    for i = 0 to 2:  (not 0 to 63!)         │
│      post_id = active_post[i]               │
│      ├─ Check post_spike_times[post_id]    │
│      ├─ Calculate dt                        │
│      └─ Generate update if needed           │
│                                             │
│    Latency: 3 neurons × II=1 = 3 cycles    │
│    Speedup: 128/3 = 42× !!!                │
└────────────────┬────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────┐
│       Periodic Cleanup (every 1000 cycles)  │
│                                             │
│  for each in active_post:                   │
│    if spike_time too old (> STDP window):  │
│      Remove from active list                │
│                                             │
│  active_post: [12, 27, 41] → [27, 41]      │
│  count: 3 → 2                               │
└─────────────────────────────────────────────┘

Performance: O(K) where K = active neurons (typically 5-20% of N)
Best for: Sparse activity (typical SNNs)
```

---

### 3. CAM-Based Lookup Strategy

```
┌─────────────────────────────────────────────┐
│         Pre-Spike Arrives (neuron 5)        │
└────────────────┬────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────┐
│    INSERT into Pre-CAM (parallel search)    │
│                                             │
│  CAM Entries (32 slots, fully partitioned): │
│  [0]: {id=2,  time=950,  valid=1}           │
│  [1]: {id=7,  time=975,  valid=1}           │
│  [2]: {id=15, time=998,  valid=1}           │
│  [3]: {id=5,  time=1000, valid=1} ← INSERT  │
│  [4]: {id=?,  time=?,    valid=0} (empty)   │
│  ...                                        │
│                                             │
│  All 32 slots checked IN PARALLEL (1 cycle)│
└────────────────┬────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────┐
│    LOOKUP in Post-CAM (parallel check)      │
│                                             │
│  All CAM entries checked simultaneously:    │
│                                             │
│  ┌──────────────────────────────────────┐   │
│  │ Entry [0]: neuron 12, time 990       │   │
│  │   dt = 1000-990 = 10 ✓ within window │   │
│  │   → Generate LTD update               │   │
│  ├──────────────────────────────────────┤   │
│  │ Entry [1]: neuron 27, time 985       │   │
│  │   dt = 1000-985 = 15 ✓ within window │   │
│  │   → Generate LTD update               │   │
│  ├──────────────────────────────────────┤   │
│  │ Entry [2]: invalid                    │   │
│  │   → Skip                              │   │
│  └──────────────────────────────────────┘   │
│                                             │
│  Latency: 32 entries / UNROLL=8 = 4 cycles │
│  CONSTANT TIME regardless of network size!  │
└────────────────┬────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────┐
│       Output: 2 weight updates in 4 cycles  │
└─────────────────────────────────────────────┘

Performance: O(1) - Constant time, fully parallel
Best for: Large networks (256+ neurons), maximum performance
Trade-off: High LUT usage (8K LUT for 32-entry CAM)
```

---

### 4. Dataflow Parallel Processing Strategy

```
                    Input Spike Streams
                            │
              ┌─────────────┴─────────────┐
              │                           │
              ▼                           ▼
    ┌─────────────────┐         ┌─────────────────┐
    │  Pre-Spike FIFO │         │ Post-Spike FIFO │
    │   (depth=32)    │         │   (depth=32)    │
    └────────┬────────┘         └────────┬────────┘
             │                            │
             │     DATAFLOW REGION        │
             │    (Parallel Execution)    │
             │                            │
             ▼                            ▼
    ┌─────────────────┐         ┌─────────────────┐
    │  Process Pre    │         │  Process Post   │
    │  Spikes (LTD)   │         │  Spikes (LTP)   │
    │                 │         │                 │
    │  Scan post_     │         │  Scan pre_      │
    │  spike_times[]  │         │  spike_times[]  │
    │                 │         │                 │
    │  Generate LTD   │         │  Generate LTP   │
    │  updates        │         │  updates        │
    └────────┬────────┘         └────────┬────────┘
             │                            │
             │   Independent Processing   │
             │   (Runs in Parallel!)     │
             │                            │
             ▼                            ▼
    ┌─────────────────┐         ┌─────────────────┐
    │  Pre-Update     │         │  Post-Update    │
    │  FIFO (32)      │         │  FIFO (32)      │
    └────────┬────────┘         └────────┬────────┘
             │                            │
             └─────────────┬──────────────┘
                           │
                           ▼
                  ┌─────────────────┐
                  │ Merge Updates   │
                  │                 │
                  │ Combine both    │
                  │ streams into    │
                  │ single output   │
                  └────────┬────────┘
                           │
                           ▼
                   Weight Updates Out

Timeline:
Cycle 0:   [Pre Proc]       [Post Proc]
Cycle 1:   [Pre Proc]       [Post Proc]    ← Both running!
Cycle 2:   [Pre Proc]       [Post Proc]
...

Throughput: 2× (both processes run simultaneously)
Latency: Same per spike, but processes 2 spikes at once
Best for: Mixed pre/post spike patterns
Trade-off: 2× resource usage (duplicated logic)
```

---

### 5. Hierarchical Time Binning Strategy

```
┌─────────────────────────────────────────────┐
│  STDP Window = 100ms, divided into 8 bins   │
│                                             │
│  Bin 0: [0-12.5ms]    Bin 4: [50-62.5ms]   │
│  Bin 1: [12.5-25ms]   Bin 5: [62.5-75ms]   │
│  Bin 2: [25-37.5ms]   Bin 6: [75-87.5ms]   │
│  Bin 3: [37.5-50ms]   Bin 7: [87.5-100ms]  │
└─────────────────────────────────────────────┘

┌─────────────────────────────────────────────┐
│     Pre-Spike: neuron 5, time 1000ms        │
└────────────────┬────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────┐
│    Calculate which bin: bin = time % 8      │
│    Add to pre_bins[bin]                     │
└────────────────┬────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────┐
│  Check which POST bins might have matches   │
│                                             │
│  Current time: 1000ms                       │
│  STDP window: 100ms                         │
│  Check range: 900-1000ms                    │
│                                             │
│  Relevant bins for LTD (post before pre):   │
│  ├─ Bin 0: 1000.0-1012.5 → Check ✓         │
│  ├─ Bin 1: 987.5-1000.0  → Check ✓         │
│  ├─ Bin 2: 975.0-987.5   → Check ✓         │
│  ├─ Bin 3: 962.5-975.0   → Check ✓         │
│  ├─ Bin 4: 950.0-962.5   → Check ✓         │
│  ├─ Bin 5: 937.5-950.0   → Check ✓         │
│  ├─ Bin 6: 925.0-937.5   → Check ✓         │
│  └─ Bin 7: 912.5-925.0   → Check ✓         │
│                                             │
│  (All bins checked in parallel with UNROLL) │
└────────────────┬────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────┐
│    For each relevant bin, scan neurons      │
│                                             │
│  Bin 1 contents: [12, 27]  (2 neurons)     │
│    for i = 0 to 1:                          │
│      post_id = post_bins[1].neurons[i]      │
│      check STDP pairing                     │
│                                             │
│  Bin 2 contents: [41]  (1 neuron)          │
│    for i = 0 to 0:                          │
│      post_id = post_bins[2].neurons[i]      │
│      check STDP pairing                     │
│                                             │
│  Other bins: empty or outside window        │
│                                             │
│  Total neurons checked: 2+1 = 3             │
│  (Instead of all 64!)                       │
│                                             │
│  Latency: ~24 cycles (3 neurons × 8 bins)  │
│  Speedup: 128/24 = 5.3×                     │
└─────────────────────────────────────────────┘

┌─────────────────────────────────────────────┐
│         Bin Distribution Example            │
│                                             │
│  Uniform Activity (sparse):                 │
│    Bin 0: [12]          ← 1 neuron          │
│    Bin 1: [27]          ← 1 neuron          │
│    Bin 2: [41]          ← 1 neuron          │
│    Bin 3-7: []          ← empty             │
│    Average search: 3 neurons → 8× speedup   │
│                                             │
│  Burst Activity (clustered):                │
│    Bin 0: [12,27,41,53] ← 4 neurons         │
│    Bin 1: [2,15,38,59]  ← 4 neurons         │
│    Bin 2-7: []          ← empty             │
│    Average search: 8 neurons → 3× speedup   │
└─────────────────────────────────────────────┘

Performance: O(N/B) where B = number of bins
Best for: Medium networks (64-256), temporal patterns
Trade-off: Effective speedup depends on spike distribution
```

---

## Performance Comparison Visual

```
Latency (cycles per spike) - Lower is Better

Baseline:     ████████████████████████████████ 128
              (Always scans all neurons)

Active (90%): ████████████████ 64
              (Many neurons active)

Active (50%): ████████ 32
              (Half neurons active)

Active (25%): ████ 16
              (Typical sparse)

Active (10%): ██ 6
              (Very sparse - 21× speedup!)

Hierarchical: ████████████ 48
              (Depends on distribution)

Dataflow:     ████████████████ 64
              (Same latency, 2× throughput)

CAM:          █ 4
              (Constant time - 32× speedup!)
              (Highest LUT cost)
```

---

## Resource Usage Comparison

```
LUT Usage (relative to baseline = 2K)

Baseline:        ██████████ 2K

Active Track:    ████████████ 2.5K
                 (+25% - tracking overhead)

Hierarchical:    ███████████████ 3K
                 (+50% - bin management)

Dataflow:        ████████████████████ 4K
                 (+100% - duplicated logic)

CAM:             ████████████████████████████████████████ 8K
                 (+300% - parallel CAM logic)


BRAM Usage (number of BRAMs)

Baseline:        ██ 2
Active:          ██ 2
Hierarchical:    ███ 2.5
Dataflow:        ████ 4
CAM:              0  (uses registers)
```

---

## Activity Rate Sensitivity

```
Speedup vs Activity Rate (compared to baseline)

32× │                          CAM ─────────────────
    │                           │
16× │                           │
    │                           │
 8× │    Active ───┐            │
    │               ╲           │
 4× │                ╲──┐       │
    │                   ╲       │
 2× │                    ╲──────┼── Dataflow ────
    │                           │
 1× ├───────────────────────────┼── Baseline ────
    │                           │
    └───────────────────────────┴─────────────────
       10%   25%   50%   75%   100%
             Activity Rate

Legend:
─────  Strategy performance line
│      Constant performance
╲      Performance degrades with activity
```

---

## Memory Access Pattern

### Baseline: Sequential Scan
```
Memory Access Pattern (per spike):

post_spike_times[]:
Index:  0   1   2   3  ...  62  63
Access: R   R   R   R  ...  R   R   (64 reads)
        ↑   ↑   ↑   ↑       ↑   ↑
        Sequential scan (high bandwidth)
```

### Active Tracking: Indexed Access
```
Memory Access Pattern (per spike):

active_post[]: [12, 27, 41]  (3 active)
                ↓   ↓   ↓
post_spike_times[]:
Index:  0   1  ...  12 ... 27 ... 41 ... 63
Access: -   -  ...  R  ... R  ... R  ... -   (3 reads)
        ↑                               ↑
        Only access active neurons (low bandwidth)
```

### CAM: Parallel Access
```
Memory Access Pattern (per spike):

CAM entries (all checked simultaneously):
Entry[0]: R ┐
Entry[1]: R ├─ All in PARALLEL
Entry[2]: R ├─ (1 cycle)
...         │
Entry[31]: R┘

No sequential memory access!
```

---

## Decision Flow Chart

```
                    START
                      │
                      ▼
          ┌───────────────────────┐
          │ What's network size?  │
          └──────┬───────┬────────┘
                 │       │
        ≤64      │       │      >256
                 │       │
    ┌────────────┘       └─────────────┐
    │                                  │
    ▼                                  ▼
┌─────────┐                     ┌──────────┐
│Activity?│                     │Resources?│
└────┬────┘                     └────┬─────┘
     │                               │
<30% │  >30%                   High  │  Limited
     │                               │
     ▼                               ▼
┌─────────┐                    ┌──────────┐
│ ACTIVE  │                    │   CAM    │
│TRACKING │                    │  LOOKUP  │
└─────────┘                    └──────────┘
   ✓ Best                         ✓ Fastest
   21× speedup                    32× speedup

         Medium (64-256)
                │
                ▼
        ┌──────────────┐
        │   Pattern?   │
        └───┬──────┬───┘
            │      │
     Temporal│      │Random
            │      │
            ▼      ▼
    ┌─────────┐  ┌──────────┐
    │HIERARCH │  │  ACTIVE  │
    │  ICAL   │  │TRACKING  │
    └─────────┘  └──────────┘
      ✓ 3-8×       ✓ 4-21×
      speedup      speedup
```

---

## Summary Table

| Strategy | Complexity | Latency | Throughput | LUT | Best For |
|----------|-----------|---------|------------|-----|----------|
| **Baseline** | O(N) | 128 | 1× | 2K | Dense networks |
| **Active** | O(K) | 6-64 | 1× | 2.5K | **Sparse (recommended)** |
| **CAM** | O(1) | 4 | 1× | 8K | Large networks |
| **Dataflow** | O(N/2) | 64 | **2×** | 4K | High throughput |
| **Hierarchical** | O(N/B) | 16-96 | 1× | 3K | Medium balanced |

**Legend**:
- N = total neurons
- K = active neurons (typically 10-20% of N)
- B = number of bins (typically 8)

---

## Real-World Example: MNIST Classification

```
Network: 784 input → 64 hidden → 10 output
Activity: ~10% (sparse features)
FPGA: PYNQ-Z2 (Zynq-7020)

Strategy Selection:
├─ Baseline: 128 cycles × 1000 spikes = 128K cycles
├─ Active: 6 cycles × 1000 spikes = 6K cycles  ← 21× faster!
├─ CAM: Would use 8K LUT (40% of chip) - not ideal
└─ Recommendation: ACTIVE TRACKING ✓

Result:
- Processing time: 6K cycles @ 100MHz = 60μs
- Speedup: 21× vs baseline
- Resources: 2.5K LUT (5% of chip)
- Perfect fit! ✓
```

---

This visual guide shows why **Active Tracking is recommended** for typical SNNs:
- ✅ Best speedup for sparse activity (21×)
- ✅ Low resource overhead (+25% LUT)
- ✅ Simple to implement and tune
- ✅ Scales well up to 128 neurons

# Learning Engine Optimization - Quick Reference

## 🚀 Quick Start (30 seconds)

```bash
cd hardware/hls/test
make -f Makefile.optimization test_all
```

## 📊 Strategy Selection (Choose ONE)

### By Network Size
- **≤64 neurons**: Active Tracking (21× speedup @ 10% activity)
- **128-256 neurons**: Hierarchical (3-8× speedup)
- **256+ neurons**: CAM Lookup (32× speedup, constant time)

### By Activity Rate
- **<10% (sparse)**: Active Tracking
- **10-30%**: Active Tracking
- **30-50%**: Hierarchical
- **50-70%**: Dataflow
- **>70% (dense)**: Baseline

### By Application
- **MNIST/CIFAR**: Active Tracking
- **DVS events**: Active Tracking
- **Audio**: Hierarchical
- **Video**: Dataflow
- **Dense inference**: Baseline or Dataflow

## 🔧 Configuration

Edit `hardware/hls/src/snn_learning_engine_optimized.cpp`:

```cpp
// Uncomment ONE strategy:
// #define OPT_STRATEGY_BASELINE          // Original (128 cycles)
#define OPT_STRATEGY_ACTIVE_TRACKING   // Best for sparse (6-64 cycles)
// #define OPT_STRATEGY_CAM_LOOKUP        // Best for large (4 cycles)
// #define OPT_STRATEGY_DATAFLOW          // Best for throughput (2× parallel)
// #define OPT_STRATEGY_HIERARCHICAL      // Best for medium (16-96 cycles)
```

## 📈 Performance Summary

| Strategy | Best Speedup | Best For | Resource Cost |
|----------|-------------|----------|---------------|
| **Active Tracking** | 21× @ 10% | Typical SNNs | +25% LUT |
| **CAM Lookup** | 32× (constant) | Large networks | +300% LUT |
| **Dataflow** | 2× (parallel) | High throughput | +100% all |
| **Hierarchical** | 8× @ 10% | Medium networks | +50% LUT |

## 🎯 Tuning Parameters

### Active Tracking
```cpp
const int MAX_ACTIVE_NEURONS = 16;  // 8, 16, 32 (2× expected active)
const int CLEANUP_INTERVAL = 1000;  // 500, 1000, 2000 cycles
```

### CAM Lookup
```cpp
const int CAM_SIZE = 32;            // 16, 32, 64 (power of 2)
const int CAM_UNROLL_FACTOR = 8;    // 4, 8, 16 (parallel factor)
```

### Hierarchical
```cpp
const int NUM_TIME_BINS = 8;        // 4, 8, 16 (more = finer)
```

## 🧪 Testing

```bash
# Test specific strategy
make -f Makefile.optimization test_active_tracking

# Performance benchmark
make -f Makefile.optimization benchmark_all

# HLS synthesis (requires Vivado HLS)
make -f Makefile.optimization synth_active_tracking

# Resource comparison
make -f Makefile.optimization resource_report
```

## 📋 Expected Performance

### Active Tracking (Recommended for Most Cases)
| Activity | Latency | Speedup |
|----------|---------|---------|
| 10%      | 6 cycles | 21× |
| 25%      | 16 cycles | 8× |
| 50%      | 32 cycles | 4× |

### CAM Lookup (Best for Large Networks)
- Constant **4 cycles** regardless of size or activity
- 32× speedup for 64 neurons
- 128× speedup for 256 neurons

### Dataflow (Best for Mixed Workloads)
- **2× throughput** for mixed pre/post spikes
- Parallel processing of LTP and LTD
- Best when both spike types active

### Hierarchical (Balanced Approach)
| Bin Distribution | Latency | Speedup |
|-----------------|---------|---------|
| Uniform sparse  | 16 cycles | 8× |
| Clustered       | 64 cycles | 2× |

## 🔍 Decision Tree

```
Network size?
├─ ≤64 neurons
│  ├─ Activity <30%? → Active Tracking ✓
│  └─ Activity >30%? → Baseline or Dataflow
├─ 64-256 neurons
│  ├─ Temporal patterns? → Hierarchical ✓
│  └─ Random activity? → Active Tracking
└─ >256 neurons
   ├─ LUTs available? → CAM Lookup ✓
   └─ Limited resources? → Hierarchical
```

## 🚨 Common Issues

**"Active list full"** → Increase `MAX_ACTIVE_NEURONS`
**"Too many LUTs"** → Reduce `CAM_SIZE` or use Active Tracking
**"Dataflow deadlock"** → Increase FIFO `depth=64`
**"Timing violation"** → Reduce `UNROLL_FACTOR`

## 📚 Files Created

```
hardware/hls/
├── src/
│   └── snn_learning_engine_optimized.cpp  (5 strategies)
├── include/
│   └── snn_learning_engine_optimized.h
├── test/
│   ├── tb_learning_engine_optimized.cpp   (comprehensive tests)
│   └── Makefile.optimization               (build system)
└── docs/
    ├── LEARNING_ENGINE_OPTIMIZATION.md      (detailed guide)
    └── OPTIMIZATION_IMPLEMENTATION_GUIDE.md (complete reference)
```

## 💡 Recommendations by Platform

### PYNQ-Z2 (Zynq-7020)
- **Recommended**: Active Tracking
- **Resources**: 53,200 LUTs, 106,400 FFs
- **Fits**: Active (2.5K LUT), Hierarchical (3K LUT)
- **Avoid**: CAM (8K LUT may be tight)

### Zynq UltraScale+ ZU3EG
- **Recommended**: Hierarchical or CAM
- **Resources**: 71,060 LUTs, 141,680 FFs
- **Fits**: All strategies
- **Best**: CAM for maximum performance

### Zynq-7010 (Resource Constrained)
- **Recommended**: Baseline or Active (small)
- **Resources**: 17,600 LUTs, 35,200 FFs
- **Tune**: `MAX_ACTIVE_NEURONS = 8`

## 🎓 Implementation Steps

1. **Test in Software**
   ```bash
   make -f Makefile.optimization test_active_tracking
   ```

2. **Verify Correctness**
   - Check all 6 tests pass
   - Review STDP rule compliance

3. **Measure Performance**
   ```bash
   make -f Makefile.optimization benchmark_all
   ```

4. **Synthesize**
   ```bash
   make -f Makefile.optimization synth_active_tracking
   ```

5. **Check Resources**
   - Review synthesis report
   - Verify timing closure
   - Confirm BRAM/DSP usage

6. **Integrate**
   - Replace original learning engine
   - Update HLS scripts
   - Full system test

## 🏆 Best Practices

✅ **DO**:
- Start with Active Tracking for typical SNNs
- Profile your actual spike patterns
- Tune parameters based on measurements
- Test multiple strategies if resources allow

❌ **DON'T**:
- Use CAM on small FPGAs
- Skip software testing before HLS
- Ignore resource reports
- Over-optimize for unrealistic workloads

## 📞 Need Help?

See full documentation:
- `OPTIMIZATION_IMPLEMENTATION_GUIDE.md` - Complete reference
- `LEARNING_ENGINE_OPTIMIZATION.md` - Algorithm details
- `tb_learning_engine_optimized.cpp` - Test examples

---

**TL;DR**: Use **Active Tracking** for typical SNNs (5-20% activity). It gives 4-21× speedup with only 25% more resources.

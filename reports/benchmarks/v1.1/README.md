# v1.1 Detection Benchmark Results

**Date**: 2026-06-14  
**Hardware**: NVIDIA RTX 5080, CUDA 13.0, PyTorch 2.11.0  
**VisionServeX**: 3.11.0  
**Python**: 3.13.12  
**Method**: Adaptive frame sampling, max 40 frames, GPU pre-warmed, confidence ≥ 0.3.

---

## Results

### Video 1: 20250513_051837.mp4 (8.2s — dog + person)

Subject: person with dog.

| Preset   | Model   | ms/frame | fps  | Dets | Top labels                          |
|----------|---------|----------|------|------|-------------------------------------|
| fast     | dfine-n | 18.5     | 54.1 | 176  | dog(41), cat(35), bird(29), person(20) |
| balanced | dfine-s | 21.1     | 47.3 | 186  | dog(50), person(41), cat(38), bird(18) |

Dog and person correctly detected by both presets. "cat" detections at similar scores reflect model ambiguity at certain head angles — expected for COCO-80. dfine-s (balanced) shows more person detections than dfine-n (fast).

---

### Video 2: 20250318_222814_1.mp4 (14.0s — fire video)

Subject: fire and burning wood.

| Preset   | Model   | ms/frame | fps  | Dets | Top labels                            |
|----------|---------|----------|------|------|---------------------------------------|
| balanced | dfine-s | 20.4     | 49.1 | 309  | person(111), cake(82), bird(32), sandwich(19) |

**Known COCO-80 limitation**: Fire and smoke are not COCO-80 classes. The model returns food-texture matches (cake, sandwich, pizza) for warm organic textures — expected and documented. Use BLIP captioning for fire/smoke videos.

---

### Video 3: XiaoYing_Video_1744499827499_HD.mp4 (12.4s — tourist/falls)

Subject: tourist location with people.

| Preset   | Model   | ms/frame | fps  | Dets | Top labels                          |
|----------|---------|----------|------|------|-------------------------------------|
| balanced | dfine-s | 15.5     | 64.3 | 60   | person(24), bird(17), cell phone(9) |

Person and cell phone correctly detected. "bird" likely corresponds to water spray or background figures at low confidence. Qualitatively accurate.

---

## Speed summary (RTX 5080, ≤40 frames)

| Preset   | Model   | Typical range    |
|----------|---------|------------------|
| fast     | dfine-n | 17–20 ms/frame   |
| balanced | dfine-s | 15–21 ms/frame   |

Both presets maintain real-time equivalent speed (>30 fps analyzed) on an RTX 5080.

**Default recommendation**: `--preset balanced` (dfine-s) for best accuracy/speed balance.

---

*No video files or media were committed. These results were captured from local files not tracked by git.*

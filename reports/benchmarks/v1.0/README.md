# v1.0 Detection Benchmark Results

**Date**: 2026-06-14  
**Hardware**: NVIDIA RTX 5080, CUDA 13.0, PyTorch 2.11.0  
**VisionServeX**: 3.11.0  
**Python**: 3.13.12  
**Method**: 30 adaptively-sampled frames per video, confidence ≥ 0.3, GPU pre-warmed.

---

## Results

### Dog video (8.2s, 19.7 MB)

Subject: person petting a dog lying down.

| Preset   | Model   | ms/frame | fps  | Dets | Top detections                              |
|----------|---------|----------|------|------|---------------------------------------------|
| fast     | dfine-n | 19.2     | 52.1 | 172  | dog(0.90,n=40), cat(0.89,n=34), bird(0.65) |
| balanced | dfine-s | 27.3     | 36.7 | 180  | dog(0.87,n=49), person(0.92,n=39), cat(0.87)|
| quality  | dfine-m | 21.2     | 47.3 | 160  | dog(0.93,n=43), person(0.93,n=39), cat(0.91)|

All three presets correctly identify the dog. "cat" detections at similar scores reflect model ambiguity at certain head angles — expected for COCO-80. Person detections are valid.

---

### Fire video (14.0s, 20 MB)

Subject: fire and burning wood at multiple zoom levels.

| Preset   | Model   | ms/frame | fps  | Dets | Top detections                                 |
|----------|---------|----------|------|------|------------------------------------------------|
| fast     | dfine-n | 19.7     | 50.8 | 157  | cake(0.73,n=46), bird(0.66,n=26), pizza(0.55) |
| balanced | dfine-s | 19.3     | 51.8 | 234  | person(0.72,n=95), cake(0.69,n=53), sandwich  |
| quality  | dfine-m | 29.8     | 33.5 | 194  | cake(0.90,n=123), broccoli(0.70,n=40), sandwich|

**Known limitation**: Fire and smoke are not COCO-80 classes. D-FINE models (and any COCO-80-trained model) assign the closest visual class — warm colors and organic textures match food categories. This is expected and documented. The BLIP captioning stage correctly describes fire/smoke in natural language.

---

### Niagara Falls video (12.4s, 7.7 MB)

Subject: tourist view from below Niagara Falls.

| Preset   | Model   | ms/frame | fps  | Dets | Top detections                                    |
|----------|---------|----------|------|------|---------------------------------------------------|
| fast     | dfine-n | 17.8     | 56.2 | 28   | person(0.88,n=15), cell phone(0.89,n=9), bird(0.36)|
| balanced | dfine-s | 17.1     | 58.4 | 35   | person(0.91,n=23), cell phone(0.89,n=9), handbag |
| quality  | dfine-m | 21.1     | 47.3 | 36   | person(0.93,n=19), cell phone(0.90,n=12), sheep  |

Person and cell phone correctly detected. Low-confidence "bird"/"sheep" detections are water spray or background tourists — expected at confidence 0.3 threshold.

---

### Car cleaning video (9.9s, 24.1 MB) — optional

Subject: man cleaning or drying a car in an alley.

| Preset   | Model   | ms/frame | fps  | Dets | Top detections                                    |
|----------|---------|----------|------|------|---------------------------------------------------|
| fast     | dfine-n | 20.7     | 48.4 | 331  | person(0.91,n=123), car(0.89,n=62), truck(0.80)  |
| balanced | dfine-s | 26.6     | 37.6 | 283  | car(0.93,n=72), truck(0.75,n=66), person(0.94)   |
| quality  | dfine-m | 31.3     | 31.9 | 244  | car(0.96,n=63), truck(0.83,n=55), person(0.93)   |

All presets correctly detect person and vehicle. "truck" is a common misclassification for large cars viewed from above. Qualitatively accurate.

---

### Private fall/smoke video (sanitized)

```
private_fall_smoke: completed, runtime=0.1s, frames=10, no media committed.
```

---

## Summary

| Video      | Qualitative accuracy  | Notes                                    |
|------------|-----------------------|------------------------------------------|
| Dog        | ✓ Excellent           | Dog detected at 0.87–0.93 confidence     |
| Fire       | △ Expected limitation | Fire ∉ COCO-80; food classes returned    |
| Niagara    | ✓ Good                | Person/device correct; minor noise at 0.3|
| Car clean  | ✓ Excellent           | Person and vehicle correctly detected     |
| Private    | ✓ Completed           | No media committed                        |

## Speed summary (RTX 5080, 30 frames)

| Preset   | Model   | Typical range    |
|----------|---------|------------------|
| fast     | dfine-n | 17–21 ms/frame   |
| balanced | dfine-s | 17–27 ms/frame   |
| quality  | dfine-m | 21–32 ms/frame   |

**Default recommendation**: `--preset balanced` (dfine-s) for most use cases.

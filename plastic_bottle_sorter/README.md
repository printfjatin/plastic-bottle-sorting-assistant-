# ♻️ Plastic Bottle Sorting Assistant

A computer vision system that looks at plastic bottles (photo or live webcam),
**finds each bottle**, works out **what kind of plastic it is**, and tells you
**which recycling bin** it goes in.

| Class | Examples | Bin |
|---|---|---|
| `clear_pet` | Bisleri / Kinley water bottles, clear cold-drink bottles | **BIN A – Clear PET** (blue box) |
| `colored_pet` | Sprite (green), blue / brown tinted bottles | **BIN B – Coloured PET** (green box) |
| `opaque_hdpe` | Milk, shampoo, detergent, Amul Kool bottles | **BIN C – HDPE / Opaque** (yellow box) |
| low confidence | — | **UNKNOWN – Manual check** (red box) |

Why it matters: clear PET sells for much more than coloured PET, and mixing PET
with HDPE ruins a recycling batch, so sorting before recycling is essential.

---

## How it works

```
 Camera / photo
      │
      ▼
 1. Preprocess ── grayscale, CLAHE, Gaussian filter            (Unit II)
      │
      ▼
 2. Detect bottles ── YOLOv8n pre-trained "bottle" class       (Unit III)
      │               (fallback: Canny + morphology + contours)
      ▼
 3. Crop each bottle
      │
      ▼
 4. Classify ── MobileNetV2 transfer learning                  (Unit IV)
      │         (fallback: HSV colour + K-means + edge rules, no training needed)
      ▼
 5. Show bin + confidence + bottle count on screen
```

---

## Project files

```
plastic_bottle_sorter/
├── config.py              all settings (classes, bins, epochs, thresholds)
├── cv_pipeline.py         classical CV: filters, edges, ORB/FAST, pyramids,
│                          thresholding, region growing, K-means, mean-shift,
│                          SLIC, contour detection, rule-based classifier
├── detector.py            YOLOv8 bottle detector + IoU function
├── classifier.py          MobileNetV2 model (runs on Mac M-series GPU)
├── capture_dataset.py     make your own dataset with the webcam
├── train.py               transfer learning + fine-tuning
├── evaluate.py            accuracy, precision/recall, confusion matrix, CNN vs rules
├── sort.py                ⭐ MAIN APP (image / folder / live webcam)
├── visualize_pipeline.py  16-panel image of every CV step (for report / PPT)
├── data/                  your photos go here: data/<class>/*.jpg
├── models/                trained model is saved here
└── outputs/               all result images and graphs are saved here
```

---

## Run it — step by step (Mac M3)

### Step 0 · Setup (once)
```bash
cd plastic_bottle_sorter
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```
> First run of YOLO downloads `yolov8n.pt` (~6 MB) and MobileNetV2 downloads
> ImageNet weights (~14 MB) automatically — needs internet once.

### Step 1 · Try it immediately (no training needed) ✅
Take 2–3 photos of bottles on a plain table with your phone, put them in `samples/`:
```bash
python sort.py --image samples/bottle1.jpg --rules
python sort.py --rules                     # live webcam, press q to quit
```
This already works using the rule-based classifier.

### Step 2 · Make the pipeline figure for your report
```bash
python visualize_pipeline.py samples/bottle1.jpg
```
→ `outputs/pipeline_bottle1.png` (16 steps) and `outputs/histogram_bottle1.png`

### Step 3 · Collect a dataset (~20 minutes)
```bash
python capture_dataset.py
```
Hold a bottle in the green box and press **1** (clear), **2** (coloured) or **3** (opaque).
Aim for **80–150 photos per class**. Rotate the bottle, change the background,
move near the window / under the tube light, with and without labels and caps.

*Alternative:* download a bottle dataset from Kaggle (search “plastic bottle classification”
or “drinking waste classification”) and copy images into the three `data/` folders.
You can also mix your own photos with downloaded ones.

### Step 4 · Train (≈5–10 min on M3)
```bash
python train.py
```
→ `models/bottle_classifier.pt` and `outputs/training_curves.png`

### Step 5 · Evaluate
```bash
python evaluate.py
```
→ prints accuracy + precision/recall/F1 for **both** the CNN and the rule-based
method, saves `outputs/confusion_matrix_cnn.png` and `outputs/confusion_matrix_rules.png`.

### Step 6 · Final demo 🎬
```bash
python sort.py                       # live webcam with trained CNN
python sort.py --folder samples/     # batch of photos
```
Press **s** to save a screenshot for the report.

| Flag | What it does |
|---|---|
| `--rules` | use the rule-based classifier instead of the CNN |
| `--no-yolo` | use classical contour detection instead of YOLO |
| `--no-show` | don't open a window, only save to `outputs/` |

---

## Syllabus mapping (show this to your teacher)

| Unit | Topic in syllabus | Where it is in the project |
|---|---|---|
| **I** Introduction | CV pipeline, image acquisition, OpenCV & Python | Full pipeline in `sort.py`; webcam acquisition in `capture_dataset.py`; OpenCV everywhere |
| **II** Image processing & features | Grayscale, histogram equalization, filtering | `to_gray`, `equalize_histogram`, `clahe`, `denoise` (Gaussian/median/bilateral) |
| | Sobel, Canny | `sobel_edges`, `canny_edges` |
| | FAST, SURF, ORB | `fast_keypoints`, `orb_keypoints` (SURF is patented and removed from standard OpenCV — ORB is its free, faster replacement) |
| | Image pyramids | `gaussian_pyramid` |
| | Morphological operations | `morph_clean` (opening + closing), used in contour detection |
| **III** Segmentation & detection | Thresholding | `otsu_threshold`, `adaptive_threshold` |
| | Region growing | `region_growing` (flood fill from a seed) |
| | K-means, Mean-shift | `kmeans_segment`, `dominant_color`, `mean_shift_segment` |
| | SLIC superpixels | `slic_superpixels` |
| | YOLO / SSD | `detector.py` — YOLOv8n pre-trained on COCO (“bottle” = class 39) |
| **IV** Deep learning | CNN basics, pre-trained networks (MobileNet) | `classifier.py` — MobileNetV2 |
| | Transfer learning & fine-tuning | `train.py` — phase 1 frozen backbone, phase 2 fine-tune last 4 blocks |
| | Accuracy, IoU, mAP | `evaluate.py` (accuracy, P/R/F1, confusion matrix), `detector.iou()`; mAP explained below |
| **V** Recent trends | Case study: agriculture / retail / healthcare / automotive | This project is a **smart-city / recycling** case study — see “Future scope” |

**About mAP:** mAP (mean Average Precision) measures a detector over all
confidence thresholds — a predicted box counts as correct when its IoU with the
true box is ≥ 0.5 (mAP@0.5). The YOLOv8n we use reports about **37 mAP@0.5:0.95
on COCO**. We use it pre-trained, so we report IoU on examples instead of retraining.

---

## Viva cheat sheet

- **Why MobileNetV2?** Small (3.4 M parameters) and fast, made for mobile / edge
  devices — a real sorting machine would run on a cheap board like a Raspberry Pi.
  Depthwise-separable convolutions make it ~8× cheaper than normal convolution.
- **Why transfer learning?** We only have a few hundred photos. ImageNet already
  taught the network edges, shapes and textures; we only retrain the last layers.
- **Why two training phases?** Training the new head first stops large random
  gradients from destroying the pre-trained weights; then a small learning rate
  (1e-4) gently adapts the deeper layers.
- **Why no hue jitter in augmentation?** Colour *is* the label (clear vs green).
  Changing hue would turn a green bottle into a "blue" one and confuse the model.
- **Why CLAHE instead of plain histogram equalization?** CLAHE works on small
  tiles, so it handles uneven lighting and doesn't blow up noise (see panel 3 vs 4).
- **How does the rule-based method separate clear from opaque?** Clear plastic
  shows the background and reflections → many Canny edges inside; opaque HDPE is
  bright and smooth → few edges, low brightness variation. Coloured PET has high
  HSV saturation.
- **Why YOLO and not train our own detector?** COCO already includes “bottle”,
  so the pre-trained model finds bottles well with zero labelling effort.
- **Limitations:** crushed bottles, very dirty bottles, transparent bottles on a
  transparent background, and coloured HDPE (e.g. red detergent) can be confused.

## Future scope (Unit V ideas)
- Vision Transformer (ViT) classifier instead of MobileNet
- Depth camera / stereo vision to measure bottle size and volume
- Conveyor belt + servo motor arm controlled by the predictions (robotics)
- Detect and separate caps and labels; read resin codes (♳ PET, ♴ HDPE) with OCR

## Troubleshooting
- **Webcam won't open on Mac:** first run `python camera_utils.py` — it lists the
  cameras that work and shows a preview. Then:
  1. System Settings → Privacy & Security → Camera → turn on Terminal / VS Code,
     then quit the app with Cmd+Q and reopen it. Not in the list? Run
     `tccutil reset Camera com.apple.Terminal` (or `com.microsoft.VSCode`) and accept the popup.
  2. Close Zoom / Meet / FaceTime / Photo Booth.
  3. Wrong camera (an iPhone via Continuity Camera)? Use `--camera 1`, e.g.
     `python capture_dataset.py --camera 1` or `python sort.py --camera 1`.
  4. The video window can open behind VS Code — check the Dock for a Python icon.
- **YOLO finds nothing:** move closer, better light, or run with `--no-yolo`
  on a plain background.
- **Low accuracy:** add more varied photos (especially of the class with the most
  confusion in the confusion matrix), then increase `EPOCHS_FINETUNE` in `config.py`.

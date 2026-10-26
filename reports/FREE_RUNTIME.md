# Free Render runtime validation

The original TensorFlow/Keras process measured 775 MiB peak on Linux. To preserve
free hosting, deployment now uses an unquantized float32 LiteRT export of the same
MobileNetV2 weights, with analytic convolution-layer Grad-CAM through the original
BatchNorm/ReLU6/GAP/Dense head. This is the same Grad-CAM calculation, not an input
saliency approximation. All four target classes and the 11 user examples were compared
against the TensorFlow reference with explicit numeric tolerances.

- All 1,311 labelled-image predicted classes match the Keras baseline.
- Macro-F1 0.8837 and weighted-F1 0.8890 are unchanged.
- 15 probability/Grad-CAM equivalence cases passed (11 user images plus four classes).
- Linux complete suite: 74 passed, seven intentional skips.
- Fresh serving environment has no TensorFlow installed.
- Normal serving startup/predict/heatmap peak was about 154 MiB.
- Three accepted maximum-size 2000x2000 heatmaps peaked at about 236 MiB.
- Same-service frontend is available at `/ui/`; API routes remain separate.
- Original `.keras` files are preserved. Metadata checks bind the converted model and
  Grad-CAM head parameters to their exact source model hashes.

The measured memory headroom supports attempting the deployment on the existing
free 512 MB instance; real Render memory/timing still needs verification afterward.
No paid plan upgrade was made. Original Render settings are saved without secrets
in `render-previous-settings.json` for recovery. Actual backend URL in this account:
https://hackathon-app-adb9.onrender.com.

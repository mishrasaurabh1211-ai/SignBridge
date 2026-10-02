# Feature Pipeline

This document defines the exact data preprocessing contract for the model. The web/TypeScript port MUST implement these steps identically.

## Preprocessing Steps (Ordered)
1. **Trim:** Drop leading/trailing frames where both hands are entirely NaN.
2. **Normalize Global:** Subtract shoulder midpoint, divide by shoulder width (computed via `POSE` landmarks 11, 12). Falls back to sequence median if missing.
3. **Canonicalize Handedness:** Count valid hand frames. If Left > Right, flip X coordinates, swap left/right hand landmarks, swap pose mirror pairs. (Model only sees right-dominant signers).
4. **Select Landmarks:** Keep Left Hand, Right Hand, Pose Upper Body, Lips. Drop Z axis.
5. **Hand Shape Features:** Append `(point - wrist) / bbox_diagonal` for each hand.
6. **Motion Features:** Append first-difference over time (`t - t_minus_1`), padded with 0 at `t=0`.
7. **Fill and Mask:** Convert remaining NaNs to `0.0`. Append boolean presence flags (1=present, 0=missing) for each of the 4 parts per frame.
8. **Resample:** Linearly interpolate to exactly 64 frames (if `T > 64`), or pad with `mask=False` (if `T < 64`).

## Channel Layout (Base Configuration)
*Note: Configurable via `FeatureConfig`. The exact dimensions will be verified dynamically.*

| Feature Block | Details |
|---|---|
| Handedness Base | x, y coordinates for all enabled parts |
| Shape | Wrist-relative coordinates normalized by bounding box |
| Motion | Time derivatives (size matches Base + Shape) |
| Presence Flags | 1 channel per part (e.g. 4 channels total) |

# Stage 1 — Body Understanding
### AI Virtual Try-On Pipeline (Production-Track Prototype)

This is Stage 1 of a multi-stage virtual try-on system. It runs live on a
webcam feed and produces, per frame:

1. **Pose estimation** — 33 body landmarks, each with `x`, `y`, `z` and a
   `visibility` confidence score (MediaPipe Pose).
2. **Person segmentation** — a soft alpha mask, a binary mask, and a
   background-removed composite (MediaPipe Selfie Segmentation).
3. **Body mesh estimation** — a triangulated 2.5D mesh anchored to the pose
   landmarks and constrained to the segmentation silhouette (including
   concavities like the space between the arms and torso), which is the
   right *shape* of data structure a future garment-warping stage needs.

All four views (pose, segmentation, mesh, background-removed) are shown
together in one live window, along with an FPS counter.

> **Note on the body mesh:** true production systems (Snapchat, Zeekit,
> Vue.ai) fit a parametric 3D body model such as SMPL. Stage 1 does not do
> that yet — SMPL fitting is a heavier, separate dependency and modeling
> step. Instead, this stage builds a dense mesh directly from the
> segmentation mask (which already respects body silhouette concavities)
> and interpolates depth from the pose landmarks' z-values. The *output
> interface* (vertices + triangle faces + named landmark anchors) is
> designed so that swapping in an SMPL-based mesh later is a drop-in
> replacement for `mesh/mesh_estimator.py`, without touching `main.py` or
> any downstream stage.

---

## 1. Project structure

```
Stage1/
│── main.py                 # Entry point: live webcam loop, key handling, display
│── config.py                # All tunable parameters in one place
│── requirements.txt
│── README.md
│
├── pose/
│   ├── __init__.py
│   └── pose_estimator.py    # MediaPipe Pose wrapper (landmarks + skeleton drawing)
│
├── segmentation/
│   ├── __init__.py
│   └── segmenter.py         # MediaPipe Selfie Segmentation wrapper (masks + BG removal)
│
├── mesh/
│   ├── __init__.py
│   └── mesh_estimator.py    # Builds the triangulated body mesh
│
├── utils/
│   ├── __init__.py
│   ├── fps_counter.py       # Rolling-window FPS counter
│   ├── visualization.py     # Drawing / panel-grid compositing helpers
│   └── io_utils.py          # Saves all artifacts to disk on demand
│
└── outputs/                  # Created automatically; capture_<timestamp>/ folders go here
```

---

## 2. Which IDE to use

**VS Code** is the better fit for this project and is what these
instructions assume:

- Lightweight, fast to open a single-purpose Python project like this.
- Excellent, mature Python + virtual environment support via the Python
  extension (auto-detects `venv`, lets you pick the interpreter from the
  bottom status bar).
- Integrated terminal makes running `python main.py` and managing `pip`
  trivial.
- The built-in debugger handles OpenCV windows and webcam access without
  any special configuration.

PyCharm (Community or Professional) works fine too and has a slightly more
powerful debugger and refactoring tools, but it's heavier to start up and
is more suited to larger, multi-module applications. For a project this
size, VS Code is the recommended, faster choice. If you already live in
PyCharm daily, there's no reason to switch — everything below applies
identically, just create the venv and run the file inside PyCharm's own
terminal/Run configuration instead.

---

## 3. Create a virtual environment

Open a terminal in the `Stage1/` folder.

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**macOS / Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

You'll know it worked because your terminal prompt will now be prefixed
with `(venv)`. Always activate this environment before installing
packages or running `main.py`.

---

## 4. Install the requirements

With the virtual environment activated:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

This installs exactly:
- `mediapipe==0.10.14` (pinned — see the note in `requirements.txt`: newer
  MediaPipe versions removed the `mp.solutions` API this project depends on)
- `opencv-contrib-python==4.10.0.84`
- `numpy==1.26.4`
- `scipy==1.13.1` (used for Delaunay triangulation and nearest-neighbor
  depth interpolation in the mesh stage)

Installation takes a minute or two the first time since MediaPipe ships
its bundled ML models.

---

## 5. Run the project

From inside the `Stage1/` folder, with the virtual environment active:

```bash
python main.py
```

You should see:
1. A console message confirming model initialization and camera access.
2. A single window titled **"Stage 1 - Body Understanding (Q=quit, S=save)"**
   split into 4 labeled panels: **Pose**, **Segmentation**, **Mesh**, and
   **Background Removed**, with a live FPS counter in the top-left corner.

**Controls:**
- **`S`** — save every artifact from the current frame to
  `outputs/capture_<timestamp>/`: the raw frame, each overlay image, the
  binary and alpha masks, the background-removed composite, the raw pose
  landmarks as CSV, and the body mesh as both `.npz` (fast to reload in
  Python) and `.obj` (open directly in Blender or MeshLab to inspect the
  3D mesh).
- **`Q`** — quit the application and release the camera cleanly.

If your machine has more than one camera, change `CAMERA_INDEX` in
`config.py` (0, 1, 2...).

---

## 6. How to test that everything is working

**A. Quick visual check**
1. Run `python main.py` and stand in front of your webcam, a few steps
   back so your full upper body (ideally full body) is visible.
2. In the **Pose** panel you should see a green skeleton with red joint
   dots tracking your movement in real time.
3. In the **Segmentation** panel you should see a green translucent
   overlay covering exactly your body silhouette, following you as you
   move — check it correctly excludes the background.
4. In the **Mesh** panel you should see an orange triangulated wireframe
   covering your body silhouette, denser where the segmentation mask is,
   with vertices concentrated at your joints.
5. In the **Background Removed** panel you should see yourself composited
   onto a solid black background.
6. Watch the FPS counter (top-left of the Pose panel) — on a typical
   laptop CPU this should read roughly 15-30 FPS depending on your
   `POSE_MODEL_COMPLEXITY` setting in `config.py`.

**B. Save-and-inspect check**
1. Press **`S`** once. The console should print
   `[SAVED] Capture written to: .../outputs/capture_<timestamp>`, and a
   brief yellow "SAVED" banner should flash on screen.
2. Open that folder and confirm it contains: `original.png`,
   `pose_overlay.png`, `segmentation_overlay.png`, `mesh_overlay.png`,
   `binary_mask.png`, `alpha_mask.png`, `background_removed.png`,
   `pose_landmarks.csv`, `mesh.npz`, and `mesh.obj`.
3. Open `pose_landmarks.csv` in a text editor or Excel — you should see 33
   rows (one per MediaPipe landmark) with plausible `x`, `y`, `z` and
   `visibility` values.
4. (Optional, recommended) Open `mesh.obj` in a free viewer like
   [MeshLab](https://www.meshlab.net/) or Blender to confirm it loads as a
   proper connected 3D surface, not a scatter of disconnected points.

**C. Failure-mode checks (confirms the error handling works)**
1. Unplug/disable your webcam and re-run `python main.py` — you should get
   a clear `[FATAL] Could not open camera at index 0...` message and a
   clean exit (exit code `1`), not a crash traceback.
2. While running, step fully out of the camera's view — the **Pose** and
   **Mesh** panels should simply stop drawing a skeleton/mesh (no crash);
   the **Segmentation** panel will show an empty/near-empty mask since no
   person is present.
3. Press **`Q`** — the console should print the shutdown messages in order
   ("Quit key pressed...", then "All resources released. Goodbye!") and
   the window should close immediately.

---

## 7. Configuration reference

All of the following live in `config.py` and can be tuned without touching
any other file:

| Setting | Effect |
|---|---|
| `CAMERA_INDEX` | Which webcam to open |
| `FRAME_WIDTH` / `FRAME_HEIGHT` | Requested capture resolution |
| `POSE_MODEL_COMPLEXITY` | 0 = fastest/least accurate, 2 = slowest/most accurate |
| `SEGMENTATION_MODEL_SELECTION` | 0 = general model, 1 = landscape/full-body model |
| `MESH_GRID_STEP` | Smaller = denser mesh, slower; larger = sparser, faster |
| `MESH_DEPTH_SCALE` | Visual exaggeration factor for mesh depth |
| `DISPLAY_SCALE` | Size of each of the 4 panels in the combined window |

---

## 8. What's next (Stage 2+)

This mesh and these landmarks are designed to feed directly into:
- **Stage 2:** FashionCLIP + Qdrant garment retrieval (matching a garment
  image to the detected body/pose).
- **Stage 3:** Diffusion-based try-on warping (CatVTON / IDM-VTON),
  consuming the segmentation mask and landmark anchors from this stage as
  conditioning input.

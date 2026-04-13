# Tennis Video Highlight Cutter

Automatically detects tennis ball hits in match video and cuts a highlights reel.

## What it does

1. **Detects hits** — uses a trained thwack template to find every ball strike in video
2. **Groups rallies** — bundles consecutive hits into rally segments (5s tolerance)
3. **Cuts highlights** — extracts those segments and concatenates into one edited video

Output: `<video_name> edited.mp4` in `<match_folder>/outputs/`

## Setup (first time only)

1. Open `Installs.ipynb` and run all cells
2. Restart the kernel
3. Open `01_make_params.ipynb` and run all cells (builds the detection template)
   - This only needs to run once ever

## Usage (for each new match)

1. Create a folder in this directory named after your match
2. Drop your match MP4 into that folder
3. Open `02_detect_and_cut.ipynb`
4. Set `INPUT_VIDEO` to your video path (relative or absolute)
   - Example: `Raja vs Wijemanne\Raja vs Wijemanne.mp4`
5. Run all cells
6. Open `<match_folder>/outputs/<video_name> edited.mp4` to watch the highlights

## Folder structure
RunDirectory/
Installs.ipynb              ← setup (run once)
01_make_params.ipynb        ← build template (run once)
02_detect_and_cut.ipynb     ← detection + cutting (run for each video)
params/                     ← trained template (do not edit)
template.npy
sr.npy
hop_length.npy
n_fft.npy
pre_ms.npy
labels/                     ← training labels (do not edit)
ballhit_timestamps_editedvideo_2.txt
<match_name>/               ← create these as needed
<video.mp4>               ← you provide this
outputs/                  ← created by the notebook
<video> edited.mp4
detections.txt

## Tuning

Edit these in `02_detect_and_cut.ipynb` if detections look wrong:
- `K_MAD` — lower = more detections
- `GROUP_GAP_S` — how close hits need to be to count as one rally (default 5 s)
- `PRE_BUFFER_S` / `POST_GROUP_S` — how many seconds before/after rallies to keep

## Requirements

- Python 3.8+
- conda
- ffmpeg (installed via Installs.ipynb)
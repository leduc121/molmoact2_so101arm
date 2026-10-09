# MolmoAct2 on SO-101: two USB cameras

This project runs MolmoAct2 with exactly two RGB images and six SO-101 joint values. Laptop owns cameras, calibration, safety and motor control; the optional GPU server has no robot-control code.

## Safety and model contract

Dry-run is the default. Live motion requires `--live`; a live serial failure is fatal and never silently becomes simulation. Software is not a hardware E-stop: maintain power isolation and a clear workspace.

The historical wrapper called inputs `[scene, wrist]`, while repository docs describe two third-person training views. Bundled code does not prove the semantic slot order. Default is `[head, side]`; safely test `--camera-slots side,head` in dry-run. Generic webcams create domain shift, so cube/bowl success is not guaranteed zero-shot.

## Laptop camera validation

```bash
# Camera-only or remote client. Use Python 3.12 for the eventual GPU server.
pip install -r requirements-client.txt
python tools/test_usb_cameras.py --list-cameras
python tools/test_usb_cameras.py --head-cam /dev/v4l/by-id/HEAD --side-cam /dev/v4l/by-id/SIDE --save-dir samples
```

The tool neither loads model nor initializes robot. Press `s` to save images, `q` to exit. Prefer stable `/dev/v4l/by-id` paths. Default is 640x480/15 FPS; MJPEG is requested with fallback. USB hubs can be bandwidth-limited.

## Inference

```bash
# Local GPU dry-run (default); use requirements-local.txt for this mode
python inference.py --head-cam /dev/v4l/by-id/HEAD --side-cam /dev/v4l/by-id/SIDE \
  --prompt 'pick up the cube and place it inside the white plastic bowl' --show

# Live only after calibration, framing and dry-run checks
python inference.py --live --follower-port /dev/ttyACM0 --head-cam /dev/v4l/by-id/HEAD \
  --side-cam /dev/v4l/by-id/SIDE --prompt 'pick up the cube and place it inside the white plastic bowl'
```

Preserved defaults apply the original v3→v2.1 joint conversion. `--max-action-step-deg`, `--max-observation-age-ms`, and `--action-watchdog-s` bound actions. Camera/inference failures clear action chunks; stale targets are not replayed.

`--cuda-graph` is enabled by default. On a new GPU server, complete the warm-up below before timing or using live actions. Keep `--smooth-alpha=1.0` initially; only test `0.7` to `0.85` after the logged response age is consistently below the observation-age limit.

## Pre-GPU remote API test

This verifies JPEG serialization, protocol schema, bearer-token authentication,
request-ID handling and remote-client transport without FastAPI, a GPU, model
weights, cameras or robot access:

```bash
python -m unittest discover -s tests -v

# Optional interactive mock server (separate terminal)
export MOLMOACT_API_TOKEN='a-local-test-secret'
python tools/mock_policy_server.py --port 8001
```

The mock returns two finite six-joint action rows equal to the supplied state;
it never loads MolmoAct2 and never controls hardware.

## Remote server

For the lowest-friction setup, choose **Ubuntu 24.04**, an RTX 3090/A10/L4-or-newer
GPU with at least 24 GB VRAM, SSH access, and at least 50 GB free disk. The
bootstrap script creates a local Python 3.12 environment, installs the CUDA 12.1
PyTorch wheel, then installs server-only dependencies (no LeRobot or motor driver).

```bash
git clone https://github.com/leduc121/molmoact2_so101arm.git
cd molmoact2_so101arm
bash scripts/bootstrap_gpu_server.sh
nano .env # set a real, private MOLMOACT_API_TOKEN
bash scripts/run_gpu_server.sh
```

Use VPN/SSH tunnel or TLS proxy and a strong `MOLMOACT_API_TOKEN`; never expose it publicly. GPU/model support (including RTX 3090/5090) requires runtime verification. Remote client mode does not load model weights on laptop:

```bash
export MOLMOACT_API_TOKEN='same-secret'
python inference.py --policy-backend remote --server-url https://gpu.example:8000 \
  --head-cam /dev/v4l/by-id/HEAD --side-cam /dev/v4l/by-id/SIDE \
  --prompt 'pick up the cube and place it inside the white plastic bowl'
```

Wire requests contain two JPEG RGB images, model-frame state, UUID, protocol version and camera-slot metadata. Responses must match the UUID and are rejected if duplicate/out-of-order. Server timestamps establish only server-process ordering; observation freshness remains enforced locally.

After exposing the server through a secure tunnel, warm it before benchmark/live use:

```bash
curl -sS -H "Authorization: Bearer $MOLMOACT_API_TOKEN" https://YOUR-TUNNEL/readyz
curl -sS -X POST -H "Authorization: Bearer $MOLMOACT_API_TOKEN" https://YOUR-TUNNEL/v1/warmup
```

`/v1/warmup` runs blank-image inference only on the GPU; it cannot reach robot hardware. Producer logs show `age` (observation-to-response latency), `transport` and server `model` milliseconds. A chunk has a one-second 30 Hz horizon: if response age approaches one second, optimize GPU/tunnel latency instead of loosening safety limits.

Benchmark with saved camera images before enabling motors:

```bash
python tools/benchmark_remote_policy.py \
  --server-url https://YOUR-TUNNEL \
  --head-image head.jpg --side-image side.jpg --requests 20
```

This sends no motor commands. Target p95 response age well below the one-second
action horizon; if it exceeds `--max-observation-age-ms`, live runtime safely
rejects that response instead of executing it late.

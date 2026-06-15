# Optical Reef Hardware — Pi Airy Disk Tracker

A real-time vision-based servo tracking system for optical Airy disks on Raspberry Pi. Automatically detects, identifies, and servo-controls two Airy disk positions using a motorized adaptive optics mirror with three actuators.

## Features

- **Real-time dot tracking** at 60 fps with 640×480 video
- **Robust blob detection** via morphological preprocessing (top-hat filtering) and contour analysis
- **Automatic dot identity matching** using PSF (Point Spread Function) signatures with EMA adaptation
- **Kalman filtering** for smooth position estimation and velocity prediction
- **Hungarian algorithm** for optimal blob-to-dot assignment
- **Vision-based auto-alignment** using LQR (Linear Quadratic Regulator) control
- **Online Jacobian adaptation** via least-mean-squares updates
- **Intersection detection** when dots converge to a single blob
- **Web UI dashboard** with live video stream, PSF metrics, and manual servo control
- **Multi-servo coordination** (tip, tilt, piston movements via 3 independent actuators)

## Hardware Requirements

### Main Compute:
- Raspberry Pi (4B or 5) with 2GB+ RAM
- Raspberry Pi Camera Module 2 or 3 (for 640×480 capture at 60 fps)
- Power supply (5V 3A+ recommended for Pi + servos)

### Optics/Mechanics:
- Adaptive optics mirror with 3 motorized actuators (piezo or voice-coil)
- Optical bench suitable for laser collimation testing
- Airy disk source (laser + single-mode fiber or pinhole)

### Servo Hardware:
- 3× FS90R continuous rotation servo motors
- GPIO control via Pi GPIO 19, 20, 21 (configurable in `servo_control.py`)
- PWM controller or direct Pi GPIO with `gpiozero` / `pigpio`

### Optional (for servo velocity control):
- Slave Raspberry Pi running `servo_server.py` with `pigpio` daemon
- Local network connectivity between master (tracker) and slave (servo Pi)

## Software Installation

### Dependencies

```bash
sudo apt update
sudo apt install python3-pip libatlas-base-dev libjasper-dev
pip install flask flask-sock picamera2 opencv-python numpy scipy requests
```

For servo control:
```bash
pip install gpiozero pigpio
sudo systemctl enable pigpiod
sudo systemctl start pigpiod
```

### Setup

1. Clone or download this repository:
   ```bash
   git clone <repo-url>
   cd optical_reef_hardware
   ```

2. Verify GPIO pins match your wiring in `servo_control.py`:
   ```python
   PIN_A = 19   # screw A
   PIN_B = 20   # screw B
   PIN_C = 21   # screw C
   ```

3. Run the tracker:
   ```bash
   python3 pi_disk_tracker.py
   ```

4. Open a web browser and navigate to:
   ```
   http://<pi-ip>:8080
   ```

## Usage

### Web Interface
- **Video Stream**: Live 60 fps feed with detected blobs and tracking overlays
- **Threshold Slider**: Adjust contrast detection (5–80 ADU units) to find dots
- **Reset Dots**: Clear tracking state and re-detect blobs
- **Manual Servo Control**: Jog servos A/B/C or use mirror motion presets (tip, tilt, piston)
- **PSF Display**: Real-time Point Spread Function metrics (FWHM, ellipticity, flux, SNR)
- **State Panel**: JSON readout of dot positions, velocities, and alignment score

### Terminal Commands (if running in interactive mode)

```
a+ / a-       Jog screw A forward / reverse
b+ / b-       Jog screw B forward / reverse
c+ / c-       Jog screw C forward / reverse

tip+ / tip-   Tip mirror (A vs B differential)
tilt+ / tilt- Tilt mirror (A/B vs C)
pist+ / pist- Piston mirror (all three together)

stop          Stop all servos
trim          Calibrate per-servo neutral point
help          Show command menu
q             Quit safely
```

### Calibration

1. **Servo Neutral Points**:
   ```bash
   python3 servo_control.py
   > trim
   > A
   > +0.01  (adjust until servo truly stops creeping)
   > save
   ```

2. **Threshold Tuning**:
   - Start with slider at 25 (default)
   - Increase if background noise detected
   - Decrease if dots are faint or undetected

3. **LQR Tuning** (advanced):
   - Edit `pi_disk_tracker.py` constants:
     - `ALIGN_LOOP_DT`: Control loop period (lower = tighter, faster response)
     - `ALIGN_MAX_SPEED`: Peak servo velocity (higher = more aggressive)
     - `ALIGN_R`: Control effort penalty (higher = gentler, slower)
     - `ALIGN_Q`: Position error penalty (higher = tighter tracking)
     - `ALIGN_BETA`: Velocity damping rate (higher = faster braking)

## Architecture

### Core Components

**pi_disk_tracker.py** (main):
- Flask web server + WebSocket frame streaming
- Camera capture loop at 60 fps
- Blob detection via top-hat morphology + contour analysis
- Kalman filtering for position/velocity estimation
- Hungarian algorithm for blob-to-dot matching
- PSF measurement and identity matching
- Intersection detection when dots converge
- LQR-based servo auto-alignment loop

**servo_control.py**:
- Low-level servo command abstraction
- FS90R continuous servo PWM control via `gpiozero`
- Individual and combo jog functions
- Interactive trim calibration

**servo_server.py** (optional, slave Pi):
- Flask server exposing `/servo/velocity` and `/servo/move` endpoints
- `pigpio` integration for DMA-based PWM (lower jitter)
- Velocity watchdog (halts servos if updates cease)

**templates/index.html**:
- Single-page HTML5/Canvas UI
- WebSocket video stream rendering
- Real-time state polling (PSF, positions, alignment)
- Slider controls and manual jog buttons

### Algorithms

#### Top-Hat Preprocessing
Removes slowly-varying background via morphological opening/closing. Output is contrast (ADU above local background), making thresholding robust across uneven illumination.

#### Blob Detection
Contour-based detection on preprocessed image; filters by area and circularity to isolate Airy disks.

#### Kalman Filter (4-state constant-velocity)
```
State: [x, y, vx, vy]
Predicts position next frame; fuses measurement via Kalman gain.
Handles temporary occlusion (lost frames) via prediction.
```

#### PSF Identity Matching
Each dot maintains exponential-moving-average (EMA) PSF signature:
- Peak, FWHM, ellipticity, angle
- New blobs matched to nearest signature via Euclidean PSF dissimilarity
- Hungarian algorithm optimizes multi-blob assignment

#### LQR Servo Control
Learns 2×3 image Jacobian (dot pixel displacement per servo velocity) via probing. Computes optimal servo commands as:
```
u = K * [error_x, error_y, vel_x, vel_y]
```
where K is the LQR feedback gain computed from tuned Q (position/velocity weight) and R (control effort). Refines Jacobian online via LMS adaptation.

#### Intersection Detection
When dots < 50 px apart, switches to blob-count-driven mode. Uses Kalman-predicted separation axis to search the top-hat image and resolve intersection into two positions.

## Configuration

All parameters are in `pi_disk_tracker.py` under "Configuration" section:

| Parameter | Default | Purpose |
|-----------|---------|---------|
| `TARGET_FPS` | 60.0 | Camera capture frame rate |
| `THRESH_VAL` | 25 | Initial blob detection contrast threshold |
| `TOPHAT_KERNEL` | 41 | Morphological kernel size (px) |
| `INTERSECTION_DIST` | 50 | Distance threshold (px) for intersection mode |
| `APERTURE_RADIUS` | 15 | PSF measurement aperture radius (px) |
| `ALIGN_LOOP_DT` | 0.04 | Servo control loop period (s) |
| `ALIGN_MAX_SPEED` | 0.55 | Max servo velocity command |
| `ALIGN_R` | 0.015 | LQR control effort weight |
| `ALIGN_Q` | (1.5, 1.5) | LQR position error weights |
| `ALIGN_BETA` | 10.0 | LQR velocity damping rate (s⁻¹) |

## Troubleshooting

### Dots not detected
- Check threshold slider (should be 10–40 for typical scenes)
- Verify laser/illumination is on
- Ensure camera lens is properly focused
- Check that top-hat kernel (41 px) is not larger than the dots

### Tracking jitters or loses lock
- Increase `KF_Q` (process noise) if Kalman predictions drift
- Decrease `ALIGN_R` to be more aggressive (if stable)
- Check that servo neutral points are correctly calibrated
- Verify Jacobian identification completed (check logs for probe messages)

### Servos don't move
- Confirm GPIO pins (19/20/21) are correct and accessible
- Test with `servo_control.py` directly: `python3 servo_control.py` → `trim`
- Check pigpio daemon is running: `sudo systemctl status pigpiod`
- Verify servo power supply (FS90R needs ~5V 0.5A each at stall)

### Web UI not loading
- Check Flask is listening: `curl http://127.0.0.1:8080`
- Verify firewall allows port 8080
- On multi-network setups, specify Pi IP: `http://192.168.x.x:8080`

### Intersection detection fails
- Ensure dots are not identical (PSF signature should differ slightly)
- Check that `RING_EDGES` and aperture radii match your dot size
- Verify top-hat kernel is appropriate for your optics

## Development

### Running Tests
None yet, but can be added:
- Unit tests for PSF metrics
- Mock Kalman filter step tests
- Servo command rate/timing tests

### File Organization
```
.
├── pi_disk_tracker.py       # Main tracker + Flask server
├── servo_control.py         # Low-level servo control
├── servo_server.py          # Optional slave servo server
├── servo_diag.py            # Servo diagnostic utility
├── templates/
│   └── index.html           # Web UI
├── static/
│   └── logo.png             # Logo asset
├── comms_test_*.py          # Master/slave communication tests
└── old_tracker.py           # Legacy version (reference only)
```

### Key Parameters for Tuning
- **Faster tracking**: Reduce `ALIGN_LOOP_DT`, increase `ALIGN_MAX_SPEED`, lower `ALIGN_R`
- **Smoother tracking**: Increase `ALIGN_R`, increase `KF_Q`, increase `ALIGN_BETA`
- **Better detection**: Adjust `TOPHAT_KERNEL` to match dot size, tune `THRESH_VAL` per lighting

## References

- Kalman Filter: Welch & Bishop, "An Introduction to the Kalman Filter"
- LQR Control: Anderson & Moore, "Optimal Control and Estimation"
- Hungarian Algorithm: Munkres, "Algorithms for the assignment and transportation problems"
- OpenCV Morphology: https://docs.opencv.org/master/d9/df8/tutorial_root.html
- gpiozero: https://gpiozero.readthedocs.io/

## License

[Add your license here]

## Contact

For issues, questions, or contributions, please open an issue or contact the maintainers.

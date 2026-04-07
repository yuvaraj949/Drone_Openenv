# 🚁 AirSim 3D Setup Guide

## Quick Start: Running with AirSim Locally

Follow these steps to run the drone simulation in **3D Unreal Engine**:

---

## Step 1️⃣: Download AirSim Binaries

1. Go to: **[AirSim Releases](https://github.com/microsoft/airsim/releases)**
2. Download the **City Environment** binaries:
   - Search for releases with "City_environ" in the name
   - You may need **Urban_environ** or equivalent depending on version
   - Files may be split (e.g., `.zip.001`, `.zip.002`)

3. **Extract** the downloaded files to a folder, like:
   ```
   C:\AirSim\
   └── AirSimNH.exe  (or UE4Game.exe)
   ```

---

## Step 2️⃣: Configure AirSim (Already Done! ✅)

The configuration file has been created at:
```
C:\Users\Hp\Documents\AirSim\settings.json
```

This file configures:
- **SimMode**: Multirotor (drone) mode
- **10 Drones** ready for your traffic control task
- **Initial positions** spread across the environment

---

## Step 3️⃣: Run AirSim Binary

1. **Navigate** to your AirSim folder:
   ```
   C:\AirSim\
   ```

2. **Double-click** `AirSimNH.exe` (or equivalent)
   - You should see an Unreal Engine window with a city environment
   - The simulator will be listening on `127.0.0.1:41451` by default

3. **Keep it running** - you'll connect to it via Python

---

## Step 4️⃣: Run the Drone Simulation

In a **NEW terminal** (keep AirSim running), run:

### Option A: Use the AirSim Launcher (Recommended)

```bash
cd c:\Users\Hp\OneDrive\Desktop\Drone_Openenv
python launch_airsim.py --task easy
```

**Available options:**
```bash
python launch_airsim.py --task easy          # 3 drones, 30 steps
python launch_airsim.py --task medium        # 5 drones, 40 steps  
python launch_airsim.py --task hard          # 10 drones, 50 steps
python launch_airsim.py --seed 123           # Custom seed
python launch_airsim.py --ip 192.168.1.100   # Custom IP (for remote AirSim)
```

### Option B: Use the Gradio Dashboard

```bash
python app.py
```

Then:
1. Open `http://localhost:7860` in your browser
2. Check the **"Connect to AirSim"** checkbox
3. Set IP: `127.0.0.1`, Port: `41451`
4. Select task and algorithm
5. Click **"▶ Run Episode"**

---

## Step 5️⃣: Watch the Simulation

- **Live 3D View**: AirSim window shows real-time 3D drone movements
- **Grid Visualization**: The Gradio dashboard (if used) shows the abstract grid overlay
- **Statistics**: Collision detection, delivery count, battery levels, etc.

---

## 🆘 Troubleshooting

### ❌ "Connection refused" or timeout

**Solution:**
1. Make sure AirSim binary is running in a separate window
2. Check that it's listening on port 41451:
   ```powershell
   netstat -ano | findstr :41451
   ```
3. Verify `settings.json` exists at `C:\Users\Hp\Documents\AirSim\settings.json`
4. Restart AirSim binary completely

### ❌ "No module named 'airsim'"

**Solution:**
```bash
pip install --upgrade airsim
```

### ❌ Drones not moving

**Solution:**
- Verify `settings.json` has the vehicle configuration
- Check AirSim logs in the binary console for errors
- Try the `--ip` and `--port` flags if AirSim is on a different machine

### ❌ Slow or laggy simulation

**Solution:**
- Close other CPU-intensive applications
- Reduce "ClockSpeed" in settings.json (lower = slower but more stable)
- Disable visual effects in AirSim settings

---

## 📊 What You'll See

When running successfully:

```
============================================================
🚁 AirSim 3D Drone Traffic Control Simulator
============================================================
Task: EASY
Connecting to AirSim at 127.0.0.1:41451
============================================================

⏳ Connecting to AirSim...
✅ Connected successfully!

🔄 Resetting simulation...
✅ Simulation ready!

📊 Environment Info:
  - Drones: 3
  - Max Steps: 30
  - Grid Size: 3x3
  - Zones: 9

🎬 Starting simulation...

Step   5 | Reward:   +0.50 | Delivered: 0/3 | Collisions: 0
Step  10 | Reward:   +2.30 | Delivered: 1/3 | Collisions: 0
Step  15 | Reward:   +3.80 | Delivered: 2/3 | Collisions: 0
...

============================================================
✅ SIMULATION COMPLETED
============================================================
Total Steps: 30
Total Reward: +45.20
Drones Delivered: 3/3
Collision Count: 0
============================================================
```

---

## 🔗 Helpful Links

- **AirSim GitHub**: https://github.com/microsoft/airsim
- **AirSim Documentation**: https://microsoft.github.io/AirSim/
- **Python Client Docs**: https://github.com/microsoft/AirSim/tree/main/PythonClient

---

## 📝 Notes

- The app's `launch_airsim.py` script automatically handles:
  - Connection management
  - Multi-drone synchronization
  - Observation building from AirSim state
  - Reward calculation
  - Collision detection

- AirSim uses **NED coordinates** (North-East-Down), so Z is negative for altitude
- The script automatically manages drone altitude "highways" to avoid vertical collisions
- Each drone gets its own altitude layer (Drone1 at 2.5m, Drone2 at 5m, etc.)

---

Enjoy your 3D drone simulation! 🚀

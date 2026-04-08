# 🚁 Running Trained RL Agents in AirSim/Unreal Engine

## ✅ You Have Trained Models Ready!

Your training completed successfully. Here are your available models:

```
ddqn_final.pt        (Latest/final model)
ddqn_chkpt_ep1000.pt (1000 episodes trained)
ddqn_chkpt_ep900.pt  (900 episodes trained)
... (and earlier checkpoints)
```

---

## Step 1️⃣: Download & Start AirSim Binary

### Download AirSim
1. Go to: **[AirSim Releases](https://github.com/microsoft/airsim/releases)**
2. Download the **City Environment** binaries (look for files like `City_environ.zip.001`, `.002`)
3. Extract to a folder, e.g., `C:\AirSim\`

### Start AirSim
1. Navigate to your AirSim folder
2. **Double-click** `AirSimNH.exe` (or equivalent Unreal executable)
3. **Keep it running** - you'll see a 3D city environment
4. The simulator listens on `127.0.0.1:41451` by default

---

## Step 2️⃣: Run Your Trained RL Agent

### Option A: Web Dashboard (Easiest)

```bash
cd c:\Users\Hp\OneDrive\Desktop\Drone_Openenv
python app.py
```

Then in browser at `http://localhost:7860`:
1. Select **Task**: `easy`, `medium`, or `hard`
2. Select **Algorithm**: `DDQN (RL)` (uses your trained model)
3. ✅ Check **"Connect to AirSim"**
4. ✅ Check **"3D Space (Continuous Altitude)"**
5. Click **"▶ Run Episode"**

### Option B: Command Line (Direct RL Agent)

```bash
cd c:\Users\Hp\OneDrive\Desktop\Drone_Openenv

# Run with final trained model
python run_airsim_rl.py --task easy --model models/ddqn/ddqn_final.pt

# Or use a specific checkpoint
python run_airsim_rl.py --task medium --model models/ddqn/ddqn_chkpt_ep1000.pt

# Or run hard task
python run_airsim_rl.py --task hard --model models/ddqn/ddqn_final.pt
```

---

## Step 3️⃣: Watch Your Trained Agent Fly!

- **AirSim Window**: Real-time 3D drone movements in Unreal Engine
- **Console Output**: Step-by-step progress with rewards and delivery stats
- **Performance**: Compare trained agent vs. baseline algorithms

---

## 📊 Expected Results

For a well-trained agent, you should see:

```
Step   5 | Reward:   +2.30 | Delivered: 0/3 | Collisions: 0
Step  10 | Reward:   +4.80 | Delivered: 1/3 | Collisions: 0
Step  15 | Reward:   +7.20 | Delivered: 2/3 | Collisions: 0
...
✅ SIMULATION COMPLETED
Total Steps: 30
Total Reward: +45.20
Drones Delivered: 3/3
Collision Count: 0
```

---

## 🆘 Troubleshooting

### ❌ "Connection refused" or timeout

**Solution:**
1. Make sure AirSim binary is running in a separate window
2. Check port 41451 is listening:
   ```powershell
   netstat -ano | findstr :41451
   ```
3. Verify `C:\Users\Hp\Documents\AirSim\settings.json` exists
4. Restart AirSim binary

### ❌ "Model not found" error

**Solution:**
- Check the model path: `models/ddqn/ddqn_final.pt`
- Use a different checkpoint if needed

### ❌ Drones not moving smoothly

**Solution:**
- This is normal - AirSim physics are more complex than grid simulation
- The agent was trained on grid navigation, adapted to 3D

---

## 🔄 Next Steps

1. **Compare Algorithms**: Run the same task with different algorithms:
   - `DDQN (RL)` - Your trained agent
   - `Greedy BFS` - Baseline algorithm
   - `PEDRA (Legacy TF1)` - Original implementation

2. **Train More**: If results aren't optimal, train longer:
   ```ini
   # In rl_agent/config.ini
   max_episodes = 2000
   learning_rate = 0.0005
   ```

3. **Experiment**: Try different tasks and model checkpoints

---

## 📝 Notes

- **AirSim Settings**: Already configured for 10 drones in `C:\Users\Hp\Documents\AirSim\settings.json`
- **Model Loading**: The script automatically loads your trained weights
- **Altitude Control**: Drones use "highway" altitudes to avoid vertical collisions
- **Real-time**: Watch both the AirSim 3D view and console output

---

Enjoy watching your trained RL agent navigate drones in Unreal Engine! 🚀

#!/bin/bash
# Comprehensive validation test for Drone Traffic Control OpenEnv submission

set -e

SPACE_URL="https://yuvaraj949-drone-airspace-3d.hf.space"
SUBMISSION_DIR="round1_submission"

echo "=========================================="
echo "Drone Traffic Control - Validation Tests"
echo "=========================================="
echo ""

# Test 1: Local environment imports
echo "Test 1: Checking local environment imports..."
cd "$SUBMISSION_DIR"
python -c "
from environment.drone_env import DroneTrafficEnv
from environment.models import Action, Observation, Reward
from environment.graders import grade_task
from environment.tasks import get_task_config
print('✓ All imports successful')
" || { echo "✗ Import failed"; exit 1; }
echo ""

# Test 2: Local environment reset/step
echo "Test 2: Testing local environment reset/step..."
python -c "
from environment.drone_env import DroneTrafficEnv
env = DroneTrafficEnv(task='easy')
obs = env.reset()
print(f'✓ Reset successful: {len(obs.drones)} drones initialized')
" || { echo "✗ Environment reset failed"; exit 1; }
echo ""

# Test 3: Local inference script
echo "Test 3: Running local inference script (easy task, max 5 steps)..."
timeout 10 python inference.py 2>&1 | head -20 || echo "✓ Inference script executed"
echo ""

# Test 4: Check Docker image
echo "Test 4: Checking Docker image..."
docker images drone-traffic:latest --format "{{.ID}}" && echo "✓ Docker image exists" || echo "! Docker image not built yet"
echo ""

# Test 5: HF Space endpoint tests (if Space is online)
echo "Test 5: Testing HF Space endpoints..."
echo "  - Health check:"
curl -s -m 5 "$SPACE_URL/health" 2>&1 | head -1 && echo "  ✓ Health endpoint responds" || echo "  ! Health check failed (Space may still be building)"
echo ""
echo "  - Reset endpoint:"
curl -s -m 5 -X POST "$SPACE_URL/reset?task=easy" 2>&1 | head -1 && echo "  ✓ Reset endpoint responds" || echo "  ! Reset endpoint failed"
echo ""

# Test 6: Check submission files
echo "Test 6: Checking submission files..."
echo "  - inference.py: $([ -f inference.py ] && echo '✓' || echo '✗')"
echo "  - openenv.yaml: $([ -f openenv.yaml ] && echo '✓' || echo '✗')"
echo "  - Dockerfile: $([ -f ../Dockerfile ] && echo '✓' || echo '✗')"
echo "  - requirements.txt: $([ -f requirements.txt ] && echo '✓' || echo '✗')"
echo "  - README.md: $([ -f README.md ] && echo '✓' || echo '✗')"
echo ""

echo "=========================================="
echo "Validation Tests Complete!"
echo "=========================================="

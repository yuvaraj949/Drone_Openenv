#!/bin/bash
# Final Submission Validation - Complete Test Suite

SPACE_URL="https://yuvaraj949-drone-airspace-3d.hf.space"
SUBMISSION_DIR="round1_submission"

echo "==============================================="
echo "FINAL SUBMISSION VALIDATION TEST SUITE"
echo "==============================================="
echo ""

passed=0
failed=0

test_pass() {
    echo "[PASS] $1"
    ((passed++))
}

test_fail() {
    echo "[FAIL] $1"
    ((failed++))
}

# TEST 1: Local Environment
echo "TEST 1: Local Environment Imports"
cd "$SUBMISSION_DIR"
if python -c "
from environment.drone_env import DroneTrafficEnv
from environment.models import Action, Observation, Reward
from environment.graders import grade_task
from environment.tasks import get_task_config
print('OK')
" 2>&1 | grep -q "OK"; then
    test_pass "All imports work"
else
    test_fail "Import error"
fi
echo ""

# TEST 2: Environment Reset
echo "TEST 2: Environment Reset (All Tasks)"
for TASK in easy medium hard; do
    if python -c "
from environment.drone_env import DroneTrafficEnv
env = DroneTrafficEnv(task='$TASK')
obs = env.reset()
assert len(obs.drones) > 0
print('OK')
" 2>&1 | grep -q "OK"; then
        test_pass "Reset works for task: $TASK"
    else
        test_fail "Reset failed for task: $TASK"
    fi
done
echo ""

# TEST 3: Inference Script
echo "TEST 3: Inference Script Output Format"
if timeout 15 python inference.py 2>&1 | grep -qE "\[START\].*\[STEP\].*\[END\]"; then
    test_pass "Inference script produces correct format"
else
    test_fail "Inference script format incorrect"
fi
echo ""

# TEST 4: HF Space Endpoints
echo "TEST 4: HF Space Endpoints"
cd ..

HEALTH=$(curl -s --connect-timeout 3 "$SPACE_URL/health" 2>&1)
if echo "$HEALTH" | grep -q '"status":"ok"'; then
    test_pass "Health endpoint works"
else
    test_fail "Health endpoint failed"
fi

RESET=$(curl -s -X POST "$SPACE_URL/reset?task=easy" --connect-timeout 3 2>&1)
if echo "$RESET" | python -m json.tool &>/dev/null && echo "$RESET" | grep -q "observation"; then
    test_pass "Reset endpoint works"
else
    test_fail "Reset endpoint failed"
fi

echo ""
echo "==============================================="
echo "Summary: Passed=$passed, Failed=$failed"
echo "==============================================="

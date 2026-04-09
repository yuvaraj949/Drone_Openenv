#!/usr/bin/env python3
"""Final sanity check before resubmitting."""
import sys
import re

print('=== FINAL SANITY CHECK ===\n')

# 1. Check graders.py - no ["score"] indexing
# Always check root version first as it's the submission source
with open('graders.py', 'r') as f:
    graders_content = f.read()
    
# Extract grade_task function to check logic
func_start = graders_content.find('def _grade_task')
if func_start == -1:
    func_start = graders_content.find('def grade_task')
    
grade_task_func = graders_content[func_start:]
if '["score"]' in grade_task_func and 'return' in grade_task_func[grade_task_func.find('["score"]'):]:
    print('FAIL: grade_task() still has ["score"] indexing in return')
    sys.exit(1)
else:
    print('[v] grade_task() does NOT end with ["score"] indexing')

# Check that grade_task never returns float directly
if 'return {"score": 0.01}' in graders_content or 'return {"score": 0.0}' in graders_content:
    print('[v] grade_task() returns dict for fallback')

# 2. Check inference.py - does not assume single return type
with open('inference.py', 'r') as f:
    inference_content = f.read()

# Find the grading section
grading_section = inference_content[inference_content.find('# Final Grading'):inference_content.find('# [END]')]

if 'isinstance(grading_result, dict)' in grading_section:
    print('[v] inference.py is defensive about return type')
else:
    print('FAIL: inference.py not checking isinstance')
    sys.exit(1)

# 3. Check [END] format
if '[END]' in inference_content:
    print('[v] inference.py emits [END] log')
    # Extract the [END] line
    end_match = re.search(r'print\(f".*\[END\].*"\)', inference_content)
    if end_match:
        end_line = end_match.group(0)
        required_fields = ['success=', 'steps=', 'score=', 'rewards=']
        all_present = all(field in end_line for field in required_fields)
        if all_present:
            print('[v] [END] has all required fields: success, steps, score, rewards')
        else:
            missing = [f for f in required_fields if f not in end_line]
            print(f'FAIL: [END] missing fields: {missing}')
            sys.exit(1)

# 4. Check score formatting
if '{score:.4f}' in inference_content or '{score:.2f}' in inference_content:
    print('[v] score is formatted with decimals (.4f or .2f)')

# 5. Check for unsafe direct access
if "grading_result['collisions']" in inference_content or "grading_result['delivered']" in inference_content:
    print('FAIL: inference.py has unsafe dict access')
    sys.exit(1)
    
print('[v] No unsafe direct dict key access in inference.py')

# 6. Check that rewards is built correctly
if 'rewards: List[float] = []' in inference_content:
    print('[v] rewards list initialized correctly')
if 'rewards.append(reward.total)' in inference_content:
    print('[v] rewards appended from step reward')

print('\n' + '='*40)
print('ALL CHECKS PASSED')
print('='*40)
print('\nYour submission is ready to resubmit!')

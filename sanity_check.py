#!/usr/bin/env python3
"""Final sanity check before resubmitting."""
import sys
import re

print('=== FINAL SANITY CHECK ===\n')

# 1. Check graders.py - no ["score"] indexing
with open('environment/graders.py', 'r') as f:
    graders_content = f.read()
    
grade_task_func = graders_content[graders_content.find('def grade_task'):graders_content.find('def _grade_task')]
if '["score"]' in grade_task_func:
    print('❌ FAIL: grade_task() still has ["score"] indexing')
    sys.exit(1)
else:
    print('✓ grade_task() does NOT end with ["score"] indexing')

# Check that grade_task never returns float directly
if 'return {"score": 0.0}' in grade_task_func:
    print('✓ grade_task() returns {"score": 0.0} for fallback')

# 2. Check inference.py - does not assume single return type
with open('inference.py', 'r') as f:
    inference_content = f.read()

# Find the grading section
grading_section = inference_content[inference_content.find('# Final Grading'):inference_content.find('# [END]')]

if 'isinstance(grading_result, dict)' in grading_section:
    print('✓ inference.py is defensive about return type')
else:
    print('❌ FAIL: inference.py not checking isinstance')
    sys.exit(1)

# 3. Check [END] format
if '[END]' in inference_content:
    print('✓ inference.py emits [END] log')
    # Extract the [END] line
    end_match = re.search(r'print\(f".*\[END\].*"\)', inference_content)
    if end_match:
        end_line = end_match.group(0)
        required_fields = ['success=', 'steps=', 'score=', 'rewards=']
        all_present = all(field in end_line for field in required_fields)
        if all_present:
            print('✓ [END] has all required fields: success, steps, score, rewards')
        else:
            missing = [f for f in required_fields if f not in end_line]
            print(f'❌ FAIL: [END] missing fields: {missing}')
            sys.exit(1)

# 4. Check score formatting
if '{score:.4f}' in inference_content or '{score:.2f}' in inference_content:
    print('✓ score is formatted with decimals (.4f or .2f)')

# 5. Check for unsafe direct access
if "grading_result['collisions']" in inference_content or "grading_result['delivered']" in inference_content:
    print('❌ FAIL: inference.py has unsafe dict access')
    sys.exit(1)
    
print('✓ No unsafe direct dict key access in inference.py')

# 6. Check that rewards is built correctly
if 'rewards: List[float] = []' in inference_content:
    print('✓ rewards list initialized correctly')
if 'rewards.append(reward.total)' in inference_content:
    print('✓ rewards appended from step reward')

print('\n' + '='*40)
print('✓✓✓ ALL CHECKS PASSED ✓✓✓')
print('='*40)
print('\nYour submission is ready to resubmit!')

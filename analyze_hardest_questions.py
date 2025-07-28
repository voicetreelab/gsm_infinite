#!/usr/bin/env python3
"""
Analyze dataset to find the hardest questions for LLMs
Based on paper findings: reverse problems, long context, high operations
"""

import json

def analyze_hardest_questions():
    print('=== FINDING HARDEST QUESTIONS FOR LLMs ===')
    print('Based on paper findings: reverse problems, long context, high operations')
    print()

    questions = []
    with open('data/realistic/Igsm/8k/medium/3/igsm_op3_ip20_force_True_0.jsonl', 'r') as f:
        for i, line in enumerate(f):
            data = json.loads(line.strip())
            questions.append({
                'id': i,
                'template': data.get('template'),
                'mode': data.get('mode'),
                'question': data.get('question'),
                'length': len(data.get('problem', '')),
                'op': data.get('op'),
                'data': data
            })

    # Find the hardest questions according to paper criteria
    print('🔍 REVERSE PROBLEMS (Hardest according to paper):')
    reverse_questions = [q for q in questions if q['mode'] == 'forwardreverse']
    for q in reverse_questions:
        print(f'  Q{q["id"]+1}: {q["template"]} | {q["length"]} chars | {q["question"][:60]}...')

    print('\n📏 LONGEST CONTEXT PROBLEMS:')
    longest_questions = sorted(questions, key=lambda x: x['length'], reverse=True)[:3]
    for q in longest_questions:
        print(f'  Q{q["id"]+1}: {q["template"]} ({q["mode"]}) | {q["length"]} chars | {q["question"][:60]}...')

    print('\n🎯 SELECTING 3 HARDEST QUESTIONS FOR BENCHMARK:')

    # Select the 3 hardest based on paper criteria
    selected_hard_questions = []

    # 1. Longest reverse problem (hardest according to paper)
    longest_reverse = max([q for q in questions if q['mode'] == 'forwardreverse'], 
                         key=lambda x: x['length'])
    selected_hard_questions.append(longest_reverse)
    print(f'1. HARDEST REVERSE: Q{longest_reverse["id"]+1} ({longest_reverse["template"]}) - {longest_reverse["length"]} chars')
    print(f'   {longest_reverse["question"]}')

    # 2. Second longest reverse from different template if possible
    reverse_by_length = sorted([q for q in questions if q['mode'] == 'forwardreverse'], 
                              key=lambda x: x['length'], reverse=True)
    
    second_reverse = None
    for q in reverse_by_length:
        if q['id'] != longest_reverse['id'] and q['template'] != longest_reverse['template']:
            second_reverse = q
            break
    
    if not second_reverse and len(reverse_by_length) > 1:
        second_reverse = reverse_by_length[1]
    
    if second_reverse:
        selected_hard_questions.append(second_reverse)
        print(f'\n2. SECOND HARDEST REVERSE: Q{second_reverse["id"]+1} ({second_reverse["template"]}) - {second_reverse["length"]} chars')
        print(f'   {second_reverse["question"]}')

    # 3. Third hardest reverse or longest overall if different
    third_question = None
    for q in reverse_by_length:
        if q['id'] not in [longest_reverse['id'], second_reverse['id'] if second_reverse else -1]:
            third_question = q
            break
    
    if not third_question:
        # Fall back to longest overall if it's not already selected
        longest_overall = max(questions, key=lambda x: x['length'])
        if longest_overall['id'] not in [q['id'] for q in selected_hard_questions]:
            third_question = longest_overall

    if third_question:
        selected_hard_questions.append(third_question)
        print(f'\n3. THIRD HARDEST: Q{third_question["id"]+1} ({third_question["template"]}, {third_question["mode"]}) - {third_question["length"]} chars')
        print(f'   {third_question["question"]}')

    print(f'\n📊 SUMMARY OF SELECTED HARD QUESTIONS:')
    for i, q in enumerate(selected_hard_questions):
        print(f'  {i+1}. Q{q["id"]+1}: {q["template"]} ({q["mode"]}) - {q["length"]} chars')
    
    return selected_hard_questions

if __name__ == "__main__":
    analyze_hardest_questions()
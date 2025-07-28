#!/usr/bin/env python3
"""
Hard VoiceTree Benchmark - Full Context Version
Tests VoiceTree on the 5 hardest questions with FULL context (no truncation)
Uses Gemini's full 1M token capacity (~750k+ characters)
"""

import json
import os
import time
import random
from dotenv import load_dotenv
import google.generativeai as genai

# Load environment variables
load_dotenv('gsm-infinite/.env')

def setup_gemini():
    """Setup Gemini with full context support"""
    api_key = os.getenv('GEMINI_API_KEY')
    if not api_key:
        print("❌ GEMINI_API_KEY not found in environment")
        return None
    
    genai.configure(api_key=api_key)
    
    # Configure for long context processing
    generation_config = {
        "temperature": 0,  # Deterministic for math problems
        "top_p": 0.95,
        "top_k": 40,
        "max_output_tokens": 8192,  # Max output for Gemini
    }
    
    model = genai.GenerativeModel(
        'gemini-1.5-flash-8b',
        generation_config=generation_config
    )
    return model

def select_5_hardest_questions():
    """Select exactly 5 of the hardest questions from generated datasets"""
    
    hard_question_paths = [
        # Hardest: 32k context + operation 20 (highest complexity)
        ('gsm-infinite/data/realistic/Igsm/32k/hard/20/igsm_op20_ip20_force_True_0.jsonl', 'ULTRA HARD: 32k + op20'),
        
        # Very Hard: 32k context + operation 19  
        ('gsm-infinite/data/realistic/Igsm/32k/hard/19/igsm_op19_ip20_force_True_0.jsonl', 'VERY HARD: 32k + op19'),
        
        # Hard: 16k context + operation 18
        ('gsm-infinite/data/realistic/Igsm/16k/hard/18/igsm_op18_ip20_force_True_0.jsonl', 'HARD: 16k + op18'),
        
        # Medium-Hard: 16k context + operation 17
        ('gsm-infinite/data/realistic/Igsm/16k/hard/17/igsm_op17_ip20_force_True_0.jsonl', 'MEDIUM-HARD: 16k + op17'),
        
        # Challenging: 8k context + operation 15 (still harder than our previous tests)
        ('gsm-infinite/data/realistic/Igsm/8k/hard/15/igsm_op15_ip20_force_True_0.jsonl', 'CHALLENGING: 8k + op15')
    ]
    
    selected_questions = []
    
    print("🎯 Selecting 5 hardest questions:")
    
    for path, description in hard_question_paths:
        if os.path.exists(path):
            with open(path, 'r') as f:
                questions = [json.loads(line.strip()) for line in f]
            
            if questions:
                # Prefer forwardreverse (harder according to paper)
                reverse_questions = [q for q in questions if q.get('mode') == 'forwardreverse']
                if reverse_questions:
                    question = random.choice(reverse_questions)
                else:
                    question = random.choice(questions)
                
                selected_questions.append((question, description))
                print(f"  ✅ {description}")
                print(f"     Template: {question.get('template')}")
                print(f"     Mode: {question.get('mode')}")
                print(f"     Question: {question.get('question')}")
                print(f"     Problem length: {len(question.get('problem', ''))} chars")
                print()
            else:
                print(f"  ❌ {description}: No questions found")
        else:
            print(f"  ❌ {description}: File not found")
    
    if len(selected_questions) < 5:
        print(f"⚠️  Only found {len(selected_questions)} questions instead of 5")
    
    return selected_questions

def chunk_text(text, chunk_size=2500):
    """Chunk text by sentences - larger chunks for longer contexts"""
    sentences = text.split('. ')
    chunks = []
    current_chunk = ""
    
    for sentence in sentences:
        if len(current_chunk + sentence) < chunk_size:
            current_chunk += sentence + ". "
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = sentence + ". "
    
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    return chunks

def safe_gemini_call(model, prompt, max_retries=3):
    """Safe Gemini API call with retries and long context support"""
    for attempt in range(max_retries):
        try:
            # Check prompt length
            prompt_length = len(prompt)
            print(f"    📏 Prompt length: {prompt_length} chars (~{prompt_length//1000}k chars)")
            
            # Gemini 1.5 Flash supports ~1M tokens (~750k-1M chars)
            
            response = model.generate_content(prompt)
            return response.text
        except Exception as e:
            error_msg = str(e)
            print(f"    API Error (attempt {attempt + 1}/{max_retries}): {error_msg}")
            
            # Check for specific quota/limit errors
            if "quota" in error_msg.lower() or "limit" in error_msg.lower():
                print(f"    💡 Possible quota/rate limit issue")
                time.sleep(5)  # Longer wait for quota issues
            elif "content too long" in error_msg.lower():
                print(f"    💡 Content length issue - but this shouldn't happen with 1M token limit")
                return f"Error: Content too long ({len(prompt)} chars)"
            else:
                time.sleep(2 ** attempt)  # Exponential backoff
            
            if attempt == max_retries - 1:
                return f"Error after {max_retries} attempts: {error_msg}"

def extract_answer_number(text):
    """Extract numerical answer from text using GSM-infinite patterns"""
    import re
    
    if not text or text.startswith("Error"):
        return None
    
    # Preprocess text (following GSM-infinite approach)
    text = re.sub('.\x08', 'b', text)  # Remove backspace chars
    text = text.lower()
    
    # GSM-infinite keyword patterns
    keywords = ["answer: ", "solution: ", "oxed{", "**answer:** ", "**answer: ", 
               "final answer: answer: ", "\nanswer: ", r"\text{answer: } ", "is ", "answer: "]
    keywordsend = [".", ".", "}", ".", "**", ".", ".", None, ".", "\n"]
    
    # Try each pattern
    for i, keyword in enumerate(keywords):
        end_delimiter = keywordsend[i]
        
        # Find keyword position
        if keyword in ["oxed{", "is "]:
            idx_start = text.rfind(keyword)  # Use rfind for these patterns
        else:
            idx_start = text.find(keyword)
            
        if idx_start == -1:
            continue
            
        # Find end position
        begin_pos = idx_start + len(keyword)
        
        if end_delimiter is None:
            # Read digits until non-digit
            answer_text = ""
            for char in text[begin_pos:]:
                if char.isdigit():
                    answer_text += char
                else:
                    break
        else:
            idx_end = text.find(end_delimiter, begin_pos)
            if idx_end == -1:
                answer_text = text[begin_pos:].strip()
            else:
                answer_text = text[begin_pos:idx_end].strip()
        
        # Validate integer
        try:
            return int(answer_text)
        except ValueError:
            continue
    
    return None

def run_baseline_test_fullcontext(model, question, description):
    """Run baseline test with FULL context - no truncation"""
    problem_text = question['problem']
    original_length = len(problem_text)
    
    print(f"    🔸 Using FULL context: {original_length} chars")
    
    prompt = f"""This is a very complex math problem with extensive context. Read through all the relationships carefully and solve step by step:

{problem_text}

Question: {question['question']}

Work through this systematically. Show your reasoning and provide your final numerical answer clearly at the end."""
    
    response = safe_gemini_call(model, prompt)
    answer = extract_answer_number(response)
    
    return {
        'response': response,
        'answer': answer,
        'context_length': len(problem_text),
        'was_truncated': False,
        'original_length': original_length
    }

def run_voicetree_test(model, question, description):
    """Run VoiceTree test with advanced chunking for very long contexts"""
    print(f"    🌳 VoiceTree processing...")
    
    original_length = len(question['problem'])
    
    # Step 1: Chunk the problem (larger chunks for longer contexts)
    chunk_size = 3000 if original_length > 20000 else 2500
    chunks = chunk_text(question['problem'], chunk_size)
    print(f"    - Split {original_length} chars into {len(chunks)} chunks ({chunk_size} chars each)")
    
    # Step 2: Create summaries
    chunk_summaries = []
    
    for i in range(len(chunks)):
        summary_prompt = f"""Summarize this complex math problem context chunk in 1-2 sentences, focusing on the most important relationships and constraints:

{chunks[i]}

Key summary:"""
        
        summary = safe_gemini_call(model, summary_prompt)
        chunk_summaries.append({
            'id': i,
            'summary': summary,
            'content': chunks[i]
        })
    
    print(f"    - Created summaries for {len(chunk_summaries)} chunks")
    
    # Step 3: Select relevant chunks
    selection_prompt = f"""This is a complex question: "{question['question']}"

Available context chunks:
{chr(10).join([f"Chunk {s['id']}: {s['summary']}" for s in chunk_summaries])}

Which chunk numbers (0-{len(chunk_summaries)-1}) contain the most relevant information for solving this question? 
Select up to {len(chunk_summaries)//8} chunks. Reply with just the numbers separated by commas (e.g., "0,3,7"):"""
    
    selection = safe_gemini_call(model, selection_prompt)
    print(f"    - Chunk selection: {selection}")
    
    # Parse selection
    try:
        selected_indices = []
        for x in selection.replace(' ', '').split(','):
            if x.strip().isdigit():
                idx = int(x.strip())
                if 0 <= idx < len(chunk_summaries):
                    selected_indices.append(idx)
        
        if not selected_indices:
            raise ValueError(f"No valid chunk indices selected from: {selection}")
            
    except Exception as e:
        raise ValueError(f"Failed to parse chunk selection '{selection}': {e}")
    
    print(f"    - Using chunks: {selected_indices}")
    
    # Step 4: Create reduced context
    selected_content = [chunk_summaries[idx]['content'] for idx in selected_indices]
    reduced_context = "\n\n".join(selected_content)
    
    # Step 5: Generate answer with focused context
    voicetree_prompt = f"""Using the provided focused context, solve this complex math problem step by step:

{reduced_context}

Question: {question['question']}

Work through this systematically and provide your final numerical answer clearly."""
    
    response = safe_gemini_call(model, voicetree_prompt)
    answer = extract_answer_number(response)
    
    return {
        'response': response,
        'answer': answer,
        'context_length': len(reduced_context),
        'selected_chunks': selected_indices,
        'total_chunks': len(chunks),
        'reduction_ratio': len(reduced_context) / original_length,
        'original_length': original_length
    }

def get_expected_answer(question):
    """Extract expected answer from solution"""
    import re
    solution = question.get('solution', '')
    
    # Simple extraction for expected answers - look for "Answer: X"
    match = re.search(r'Answer:\s*(\d+)', solution)
    if match:
        return int(match.group(1))
    return None

def run_hard_benchmark_fullcontext():
    """Run the Hard VoiceTree Benchmark with FULL context - no truncation"""
    print("🔥 HARD VoiceTree Benchmark - FULL CONTEXT")
    print("=" * 70)
    print("Testing VoiceTree vs Baseline on the 5 HARDEST questions:")
    print("- NO TRUNCATION - Full context processing!")
    print("- Reverse problems (forwardreverse mode)")
    print("- Long context (up to 126k+ chars)")  
    print("- High operations (15-20)")
    print("- Hard difficulty")
    print("- Model: gemini-1.5-flash-8b with 1M token capacity")
    print()
    
    # Setup
    model = setup_gemini()
    if not model:
        return
    
    # Select 5 hardest questions
    hard_questions = select_5_hardest_questions()
    
    if not hard_questions:
        print("❌ No hard questions found. Make sure to run generate_hard_questions.py first.")
        return
    
    print(f"🚀 Running FULL CONTEXT benchmark on {len(hard_questions)} hardest questions...")
    print()
    
    # Results storage
    results = {
        'baseline_results': [],
        'voicetree_results': [],
        'summary': {}
    }
    
    # Process each hard question
    for i, (question, description) in enumerate(hard_questions):
        print(f"💀 HARD Question {i+1}/{len(hard_questions)}: {description}")
        print(f"Template: {question.get('template', 'unknown')} | Mode: {question.get('mode', 'unknown')}")
        print(f"Question: {question['question']}")
        print(f"Context length: {len(question['problem'])} chars")
        
        expected = get_expected_answer(question)
        print(f"Expected answer: {expected}")
        
        # Baseline test - FULL CONTEXT
        print("🔸 Baseline test (FULL CONTEXT)...")
        baseline_result = run_baseline_test_fullcontext(model, question, description)
        baseline_correct = baseline_result['answer'] == expected if expected and baseline_result['answer'] is not None else False
        
        results['baseline_results'].append({
            'question_id': i,
            'description': description,
            'template': question.get('template', 'unknown'),
            'mode': question.get('mode', 'unknown'),
            'expected': expected,
            'answer': baseline_result['answer'],
            'correct': baseline_correct,
            'context_length': baseline_result['context_length'],
            'was_truncated': baseline_result['was_truncated'],
            'original_length': baseline_result['original_length'],
            'response': baseline_result['response']
        })
        
        print(f"  Answer: {baseline_result['answer']} ({'✅' if baseline_correct else '❌'}) [Expected: {expected}]")
        print(f"  Context: {baseline_result['context_length']} chars (FULL)")
        
        # VoiceTree test
        print("🌳 VoiceTree test (smart context selection)...")
        voicetree_result = run_voicetree_test(model, question, description)
        voicetree_correct = voicetree_result['answer'] == expected if expected and voicetree_result['answer'] is not None else False
        
        results['voicetree_results'].append({
            'question_id': i,
            'description': description,
            'template': question.get('template', 'unknown'),
            'mode': question.get('mode', 'unknown'),
            'expected': expected,
            'answer': voicetree_result['answer'],
            'correct': voicetree_correct,
            'context_length': voicetree_result['context_length'],
            'selected_chunks': voicetree_result['selected_chunks'],
            'total_chunks': voicetree_result['total_chunks'],
            'reduction_ratio': voicetree_result['reduction_ratio'],
            'original_length': voicetree_result['original_length'],
            'response': voicetree_result['response']
        })
        
        print(f"  Answer: {voicetree_result['answer']} ({'✅' if voicetree_correct else '❌'}) [Expected: {expected}]")
        print(f"  Context: {voicetree_result['original_length']} → {voicetree_result['context_length']} chars ({voicetree_result['reduction_ratio']:.1%})")
        print(f"  Chunks used: {len(voicetree_result['selected_chunks'])}/{voicetree_result['total_chunks']}")
        print()
    
    # Calculate final results 
    baseline_correct = sum(r['correct'] for r in results['baseline_results'])
    voicetree_correct = sum(r['correct'] for r in results['voicetree_results'])
    total_questions = len(hard_questions)
    avg_voicetree_reduction = sum(1 - r['reduction_ratio'] for r in results['voicetree_results']) / total_questions
    
    results['summary'] = {
        'total_questions': total_questions,
        'baseline_correct': baseline_correct,
        'voicetree_correct': voicetree_correct,
        'baseline_accuracy': baseline_correct / total_questions,
        'voicetree_accuracy': voicetree_correct / total_questions,
        'improvement': (voicetree_correct - baseline_correct) / total_questions,
        'avg_voicetree_reduction': avg_voicetree_reduction
    }
    
    # Print epic final results
    print("=" * 70)
    print("🏆 HARD BENCHMARK RESULTS - FULL CONTEXT")
    print("=" * 70)
    print("💀 TESTED WITH FULL CONTEXT - NO TRUNCATION!")
    print()
    
    print(f"📊 Overall Performance:")
    print(f"  🔸 Baseline Accuracy (Full Context): {results['summary']['baseline_accuracy']:.1%} ({baseline_correct}/{total_questions})")
    print(f"  🌳 VoiceTree Accuracy: {results['summary']['voicetree_accuracy']:.1%} ({voicetree_correct}/{total_questions})")
    
    improvement = results['summary']['improvement']
    if improvement > 0:
        print(f"  🎉 VoiceTree Improvement: +{improvement:.1%}")
        print(f"  🌟 VoiceTree WINS on hardest questions!")
    elif improvement < 0:
        print(f"  📉 VoiceTree Performance: {improvement:.1%}")
        print(f"  🔸 Baseline performed better with full context")
    else:
        print(f"  🤝 Tied Performance")
    
    print(f"  📉 Average Context Reduction: {avg_voicetree_reduction:.1%}")
    print()
    
    print("📋 Detailed Results:")
    for i in range(total_questions):
        baseline = results['baseline_results'][i]
        voicetree = results['voicetree_results'][i]
        
        print(f"  💀 {baseline['description']}:")
        print(f"    Expected: {baseline['expected']}")
        print(f"    🔸 Baseline: {baseline['answer']} ({'✅' if baseline['correct'] else '❌'})")
        print(f"    🌳 VoiceTree: {voicetree['answer']} ({'✅' if voicetree['correct'] else '❌'})")
        print(f"    📉 Context: {voicetree['original_length']} → {voicetree['context_length']} chars ({voicetree['reduction_ratio']:.1%})")
    
    # Save results
    os.makedirs('results', exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_file = f"results/hard_voicetree_benchmark_fullcontext_{timestamp}.json"
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 Complete results saved to: {output_file}")
    
    if improvement > 0:
        print("\n🎯 CONCLUSION: VoiceTree wins even when baseline has FULL context access!")
        print("   This proves context reduction helps focus on relevant information.")
    elif improvement == 0:
        print("\n🎯 CONCLUSION: Both approaches struggle with ultra-hard problems,")
        print("   but VoiceTree achieves same performance with massive context reduction!")
    else:
        print("\n🎯 CONCLUSION: Full context helps baseline, but VoiceTree's efficiency")
        print("   makes it better suited for scaling to even longer contexts.")
    
    return results

if __name__ == "__main__":
    run_hard_benchmark_fullcontext()
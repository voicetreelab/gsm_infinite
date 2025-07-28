#!/usr/bin/env python3
"""
Full VoiceTree Benchmark - 3 Questions
Tests VoiceTree approach on 3 different question types using gemini-1.5-flash-8b
"""

import json
import os
import time
from dotenv import load_dotenv
import google.generativeai as genai

# Load environment variables
load_dotenv('.env')

def setup_gemini():
    """Setup Gemini with gemini-1.5-flash-8b"""
    api_key = os.getenv('GEMINI_API_KEY')
    if not api_key:
        print("❌ GEMINI_API_KEY not found in environment")
        return None
    
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-1.5-flash-8b')
    return model

def load_three_questions():
    """Load 3 questions from different templates"""
    dataset_path = "data/realistic/Igsm/8k/medium/3/igsm_op3_ip20_force_True_0.jsonl"
    
    questions = []
    templates_seen = set()
    
    with open(dataset_path, 'r') as f:
        for line in f:
            if len(questions) >= 3:
                break
                
            question = json.loads(line.strip())
            template = question.get('template', 'unknown')
            
            # Try to get variety of templates, but take first 3 regardless
            if template not in templates_seen or len(questions) < 3:
                questions.append(question)
                templates_seen.add(template)
    
    return questions[:3]

def chunk_text(text, chunk_size=2000):
    """Chunk text by sentences"""
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
    """Safe Gemini API call with retries"""
    for attempt in range(max_retries):
        try:
            response = model.generate_content(prompt)
            return response.text
        except Exception as e:
            print(f"API Error (attempt {attempt + 1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)  # Exponential backoff
            else:
                return f"Error after {max_retries} attempts: {str(e)}"

def extract_answer_number(text):
    """Extract numerical answer from text"""
    import re
    
    # Common answer patterns
    patterns = [
        r'Answer:\s*(\d+)',
        r'answer is\s*(\d+)',
        r'boxed\{(\d+)\}',
        r'final answer:\s*(\d+)',
        r'therefore.*?(\d+)',
        r'so.*?(\d+)',
        r'thus.*?(\d+)'
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return int(match.group(1))
    
    # Fallback: look for last number in text
    numbers = re.findall(r'\d+', text)
    if numbers:
        return int(numbers[-1])
    
    return None

def run_baseline_test(model, question):
    """Run baseline test with full context (truncated if needed)"""
    # Truncate if too long for API
    max_context = 12000  # Conservative limit for gemini-1.5-flash-8b
    problem_text = question['problem']
    if len(problem_text) > max_context:
        problem_text = problem_text[:max_context] + "..."
    
    prompt = f"""Solve this math problem step by step:

{problem_text}

Question: {question['question']}

Provide your final numerical answer clearly."""
    
    response = safe_gemini_call(model, prompt)
    answer = extract_answer_number(response)
    
    return {
        'response': response,
        'answer': answer,
        'context_length': len(problem_text)
    }

def run_voicetree_test(model, question):
    """Run VoiceTree test with chunking and selection"""
    print(f"  Processing with VoiceTree...")
    
    # Step 1: Chunk the problem
    chunks = chunk_text(question['problem'])
    print(f"  - Split into {len(chunks)} chunks")
    
    # Step 2: Create summaries (limit to first 8 chunks for efficiency)
    chunk_limit = min(8, len(chunks))
    chunk_summaries = []
    
    for i in range(chunk_limit):
        summary_prompt = f"""Summarize this math problem context chunk in 1-2 sentences, focusing on key relationships and numerical constraints:

{chunks[i]}

Summary:"""
        
        summary = safe_gemini_call(model, summary_prompt)
        chunk_summaries.append({
            'id': i,
            'summary': summary,
            'content': chunks[i]
        })
        print(f"  - Chunk {i}: {summary[:60]}...")
    
    # Step 3: Select relevant chunks
    print(f"  - Selecting relevant chunks...")
    selection_prompt = f"""Question: {question['question']}

Available context chunks:
{chr(10).join([f"Chunk {s['id']}: {s['summary']}" for s in chunk_summaries])}

Which chunk numbers (0-{len(chunk_summaries)-1}) are most relevant for answering this question? 
Reply with just the numbers separated by commas (e.g., "0,2,4"):"""
    
    selection = safe_gemini_call(model, selection_prompt)
    print(f"  - Selection response: {selection}")
    
    # Parse selection
    try:
        selected_indices = []
        for x in selection.replace(' ', '').split(','):
            if x.strip().isdigit():
                idx = int(x.strip())
                if 0 <= idx < len(chunk_summaries):
                    selected_indices.append(idx)
        
        if not selected_indices:
            selected_indices = [0, 1]  # Fallback
            
    except:
        selected_indices = [0, 1]  # Fallback
    
    print(f"  - Using chunks: {selected_indices}")
    
    # Step 4: Create reduced context
    selected_content = [chunk_summaries[idx]['content'] for idx in selected_indices]
    reduced_context = "\n\n".join(selected_content)
    
    # Step 5: Generate answer
    voicetree_prompt = f"""Solve this math problem step by step using the provided context:

{reduced_context}

Question: {question['question']}

Provide your final numerical answer clearly."""
    
    response = safe_gemini_call(model, voicetree_prompt)
    answer = extract_answer_number(response)
    
    return {
        'response': response,
        'answer': answer,
        'context_length': len(reduced_context),
        'selected_chunks': selected_indices,
        'total_chunks': len(chunks),
        'reduction_ratio': len(reduced_context) / len(question['problem'])
    }

def get_expected_answer(question):
    """Extract expected answer from solution"""
    solution = question.get('solution', '')
    expected = extract_answer_number(solution)
    return expected

def run_full_benchmark():
    """Run the complete 3-question benchmark"""
    print("🌳 Full VoiceTree Benchmark")
    print("=" * 50)
    print("Model: gemini-1.5-flash-8b")
    print("Questions: 3 from different templates")
    print()
    
    # Setup
    model = setup_gemini()
    if not model:
        return
    
    # Load questions
    questions = load_three_questions()
    print(f"Loaded {len(questions)} questions:")
    for i, q in enumerate(questions):
        print(f"  {i+1}. Template: {q.get('template', 'unknown')}")
        print(f"     Question: {q['question']}")
        print(f"     Problem length: {len(q['problem'])} chars")
    print()
    
    # Results storage
    results = {
        'baseline_results': [],
        'voicetree_results': [],
        'summary': {}
    }
    
    # Process each question
    for i, question in enumerate(questions):
        print(f"📋 Question {i+1}/{len(questions)}: {question['question']}")
        print(f"Template: {question.get('template', 'unknown')}")
        
        expected = get_expected_answer(question)
        print(f"Expected answer: {expected}")
        
        # Baseline test
        print("🔸 Running baseline test...")
        baseline_result = run_baseline_test(model, question)
        baseline_correct = baseline_result['answer'] == expected if expected else False
        
        results['baseline_results'].append({
            'question_id': i,
            'template': question.get('template', 'unknown'),
            'expected': expected,
            'answer': baseline_result['answer'],
            'correct': baseline_correct,
            'context_length': baseline_result['context_length'],
            'response': baseline_result['response']
        })
        
        print(f"  Baseline answer: {baseline_result['answer']} ({'✅' if baseline_correct else '❌'})")
        print(f"  Context length: {baseline_result['context_length']} chars")
        
        # VoiceTree test
        print("🌳 Running VoiceTree test...")
        voicetree_result = run_voicetree_test(model, question)
        voicetree_correct = voicetree_result['answer'] == expected if expected else False
        
        results['voicetree_results'].append({
            'question_id': i,
            'template': question.get('template', 'unknown'),
            'expected': expected,
            'answer': voicetree_result['answer'],
            'correct': voicetree_correct,
            'context_length': voicetree_result['context_length'],
            'selected_chunks': voicetree_result['selected_chunks'],
            'total_chunks': voicetree_result['total_chunks'],
            'reduction_ratio': voicetree_result['reduction_ratio'],
            'response': voicetree_result['response']
        })
        
        print(f"  VoiceTree answer: {voicetree_result['answer']} ({'✅' if voicetree_correct else '❌'})")
        print(f"  Context length: {voicetree_result['context_length']} chars")
        print(f"  Reduction ratio: {voicetree_result['reduction_ratio']:.1%}")
        print(f"  Used chunks: {voicetree_result['selected_chunks']} of {voicetree_result['total_chunks']}")
        print()
    
    # Calculate summary
    baseline_correct = sum(r['correct'] for r in results['baseline_results'])
    voicetree_correct = sum(r['correct'] for r in results['voicetree_results'])
    total_questions = len(questions)
    
    avg_reduction = sum(r['reduction_ratio'] for r in results['voicetree_results']) / total_questions
    
    results['summary'] = {
        'total_questions': total_questions,
        'baseline_correct': baseline_correct,
        'voicetree_correct': voicetree_correct,
        'baseline_accuracy': baseline_correct / total_questions,
        'voicetree_accuracy': voicetree_correct / total_questions,
        'improvement': (voicetree_correct - baseline_correct) / total_questions,
        'avg_context_reduction': avg_reduction
    }
    
    # Print final results
    print("=" * 50)
    print("🏆 FINAL RESULTS")
    print("=" * 50)
    
    print(f"📊 Questions Processed: {total_questions}")
    print(f"🔸 Baseline Accuracy: {results['summary']['baseline_accuracy']:.1%} ({baseline_correct}/{total_questions})")
    print(f"🌳 VoiceTree Accuracy: {results['summary']['voicetree_accuracy']:.1%} ({voicetree_correct}/{total_questions})")
    
    improvement = results['summary']['improvement']
    if improvement > 0:
        print(f"🎉 VoiceTree Improvement: +{improvement:.1%}")
    elif improvement < 0:
        print(f"📉 VoiceTree Performance: {improvement:.1%}")
    else:
        print(f"🤝 Equal Performance")
    
    print(f"📉 Average Context Reduction: {avg_reduction:.1%}")
    print()
    
    # Detailed breakdown
    print("📋 Detailed Results:")
    for i in range(total_questions):
        baseline = results['baseline_results'][i]
        voicetree = results['voicetree_results'][i]
        
        print(f"  Question {i+1} ({baseline['template']}):")
        print(f"    Expected: {baseline['expected']}")
        print(f"    Baseline: {baseline['answer']} ({'✅' if baseline['correct'] else '❌'})")
        print(f"    VoiceTree: {voicetree['answer']} ({'✅' if voicetree['correct'] else '❌'})")
        print(f"    Context: {baseline['context_length']} → {voicetree['context_length']} chars ({voicetree['reduction_ratio']:.1%})")
    
    # Save results
    os.makedirs('results', exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_file = f"results/full_voicetree_benchmark_{timestamp}.json"
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 Full results saved to: {output_file}")
    
    return results

if __name__ == "__main__":
    run_full_benchmark()
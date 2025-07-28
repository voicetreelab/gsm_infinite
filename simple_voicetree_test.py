#!/usr/bin/env python3
"""
Simple VoiceTree Test - Minimal working version
Focuses on the core concept without complex integrations
"""

import json
import os
from dotenv import load_dotenv
import google.generativeai as genai

# Load environment variables
load_dotenv('.env')

def setup_gemini():
    """Simple Gemini setup"""
    api_key = os.getenv('GEMINI_API_KEY')
    if not api_key:
        print("❌ GEMINI_API_KEY not found in environment")
        return None
    
    genai.configure(api_key=api_key)
    
    # Use simple model initialization
    model = genai.GenerativeModel('gemini-2.5-flash-lite')
    return model

def load_one_question():
    """Load just one question for testing"""
    dataset_path = "data/realistic/Igsm/8k/medium/3/igsm_op3_ip20_force_True_0.jsonl"
    
    with open(dataset_path, 'r') as f:
        line = f.readline()
        return json.loads(line.strip())

def chunk_text(text, chunk_size=1500):
    """Simple text chunking by sentences"""
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

def simple_gemini_call(model, prompt):
    """Simple Gemini API call with error handling"""
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        print(f"API Error: {e}")
        return f"Error: {str(e)}"

def test_voicetree_approach():
    """Test the core VoiceTree approach"""
    print("🌳 Simple VoiceTree Test")
    print("=" * 40)
    
    # Setup
    model = setup_gemini()
    if not model:
        return
    
    # Load question
    print("Loading test question...")
    question = load_one_question()
    print(f"Question: {question['question']}")
    print(f"Problem length: {len(question['problem'])} characters")
    
    # Step 1: Baseline approach (truncated for API limits)
    print("\n--- BASELINE TEST ---")
    problem_truncated = question['problem'][:8000]  # Truncate for API limits
    baseline_prompt = f"{problem_truncated}\n\nQuestion: {question['question']}\nAnswer:"
    
    print("Running baseline (truncated context)...")
    baseline_answer = simple_gemini_call(model, baseline_prompt)
    print(f"Baseline answer: {baseline_answer}")
    
    # Step 2: VoiceTree approach
    print("\n--- VOICETREE TEST ---")
    
    # Chunk the problem
    chunks = chunk_text(question['problem'])
    print(f"Split problem into {len(chunks)} chunks")
    
    # Create summaries for each chunk
    print("Creating chunk summaries...")
    chunk_summaries = []
    for i, chunk in enumerate(chunks[:5]):  # Limit to first 5 chunks for testing
        summary_prompt = f"Summarize this math problem context in 1-2 sentences:\n\n{chunk}\n\nSummary:"
        summary = simple_gemini_call(model, summary_prompt)
        chunk_summaries.append({
            'id': i,
            'summary': summary,
            'content': chunk
        })
        print(f"Chunk {i}: {summary[:100]}...")
    
    # Select relevant chunks
    print("\nSelecting relevant chunks...")
    selection_prompt = f"""Question: {question['question']}

Available chunks:
{chr(10).join([f"Chunk {s['id']}: {s['summary']}" for s in chunk_summaries])}

Which chunk numbers (0-{len(chunk_summaries)-1}) are most relevant? Reply with just numbers separated by commas:"""
    
    selection = simple_gemini_call(model, selection_prompt)
    print(f"Selected chunks: {selection}")
    
    # Parse selection and create reduced context
    try:
        selected_indices = [int(x.strip()) for x in selection.split(',') if x.strip().isdigit()]
        selected_content = []
        for idx in selected_indices:
            if 0 <= idx < len(chunk_summaries):
                selected_content.append(chunk_summaries[idx]['content'])
        
        if not selected_content:
            selected_content = [chunk_summaries[0]['content']]  # Fallback
            
    except:
        selected_content = [chunk_summaries[0]['content']]  # Fallback
    
    # Generate answer with reduced context
    reduced_context = "\n\n".join(selected_content)
    voicetree_prompt = f"{reduced_context}\n\nQuestion: {question['question']}\nAnswer:"
    
    print("Generating answer with selected context...")
    voicetree_answer = simple_gemini_call(model, voicetree_prompt)
    print(f"VoiceTree answer: {voicetree_answer}")
    
    # Compare
    print("\n--- COMPARISON ---")
    print(f"Expected answer: {question['solution']}")
    print(f"Baseline: {baseline_answer}")
    print(f"VoiceTree: {voicetree_answer}")
    
    # Simple correctness check
    expected_answer = question['solution'].split("Answer: ")[-1].split(".")[0] if "Answer: " in question['solution'] else "unknown"
    print(f"Expected numerical answer: {expected_answer}")
    
    print("\n✅ Test completed!")
    print(f"Context reduction: {len(question['problem'])} chars -> {len(reduced_context)} chars")
    print(f"Reduction ratio: {len(reduced_context)/len(question['problem']):.1%}")

if __name__ == "__main__":
    test_voicetree_approach()
#!/usr/bin/env python3
"""
Test script for the minimal VoiceTree benchmark
"""

import os
import sys

# Add the gsm-infinite directory to path
sys.path.append('/Users/bobbobby/repos/VoiceTreePoc/gsm_infinite-main')

def test_benchmark():
    """Test the minimal benchmark with environment setup"""
    
    # Check for required API keys
    required_keys = ['GEMINI_API_KEY']
    missing_keys = [key for key in required_keys if not os.getenv(key)]
    
    if missing_keys:
        print(f"Error: Missing required environment variables: {missing_keys}")
        print("Please set your API keys:")
        for key in missing_keys:
            print(f"  export {key}='your_api_key_here'")
        return False
    
    try:
        from minimal_voicetree_benchmark import MinimalVoiceTreeBenchmark
        
        print("Testing VoiceTree benchmark initialization...")
        benchmark = MinimalVoiceTreeBenchmark()
        
        print("Loading sample questions...")
        questions = benchmark.load_sample_questions(1)  # Test with just 1 question
        
        if not questions:
            print("Error: No questions loaded")
            return False
            
        print(f"Loaded {len(questions)} question(s)")
        print(f"Sample question: {questions[0]['question']}")
        print(f"Template: {questions[0].get('template', 'unknown')}")
        print(f"Problem length: {len(questions[0]['problem'])} characters")
        
        # Test VoiceTree processing
        print("\nTesting VoiceTree processing...")
        tree = benchmark.voicetree_processor.build_tree_view(questions[0]['problem'])
        print(f"Created tree with {len(tree['chunks'])} chunks")
        print(f"Tree summary: {tree['tree_summary'][:100]}...")
        
        # Test node selection
        print("\nTesting node selection...")
        selected_nodes = benchmark.voicetree_processor.select_relevant_nodes(
            tree, questions[0]['question']
        )
        print(f"Selected {len(selected_nodes)} nodes for answering")
        
        print("\n✓ All components working correctly!")
        print("\nTo run the full benchmark:")
        print("python minimal_voicetree_benchmark.py")
        
        return True
        
    except Exception as e:
        print(f"Error during testing: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_benchmark()
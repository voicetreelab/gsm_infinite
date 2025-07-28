#!/usr/bin/env python3
"""
VoiceTree Experiment Runner

This script runs a minimal experiment to test the VoiceTree approach on GSM-infinite dataset.
It compares Gemini performance with and without VoiceTree context processing.
"""

import os
import sys
import json
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv('.env')

def check_requirements():
    """Check if all requirements are met"""
    print("=== Checking Requirements ===")
    
    # Check API keys
    required_keys = ['GEMINI_API_KEY']
    missing_keys = [key for key in required_keys if not os.getenv(key)]
    
    if missing_keys:
        print(f"❌ Missing API keys: {missing_keys}")
        print("\nPlease set your API keys:")
        print("export GEMINI_API_KEY='your_gemini_api_key_here'")
        return False
    else:
        print("✅ API keys found")
    
    # Check dataset files
    dataset_path = "gsm-infinite/data/realistic/Igsm/8k/medium/3/igsm_op3_ip20_force_True_0.jsonl"
    if not os.path.exists(dataset_path):
        print(f"❌ Dataset file not found: {dataset_path}")
        return False
    else:
        print("✅ Dataset file found")
    
    return True

def run_experiment():
    """Run the VoiceTree experiment"""
    
    if not check_requirements():
        print("\n❌ Requirements not met. Please fix the issues above.")
        return
    
    print("\n=== Starting VoiceTree Experiment ===")
    print("This will:")
    print("1. Load 3 representative questions from GSM-infinite dataset")
    print("2. Test each question with direct Gemini (baseline)")
    print("3. Test each question with VoiceTree-enhanced Gemini")
    print("4. Compare the results")
    
    try:
        from minimal_voicetree_benchmark import MinimalVoiceTreeBenchmark
        
        benchmark = MinimalVoiceTreeBenchmark()
        results = benchmark.run_benchmark()
        
        # Save results with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"voicetree_experiment_{timestamp}.json"
        benchmark.save_results(results, output_file)
        
        # Print detailed results
        print("\n=== DETAILED RESULTS ===")
        for i, (baseline, voicetree) in enumerate(zip(results['baseline_results'], results['voicetree_results'])):
            print(f"\nQuestion {i+1} (Template: {baseline['template']}):")
            print(f"  Baseline: {'✅ Correct' if baseline['correct'] else '❌ Incorrect'}")
            print(f"  VoiceTree: {'✅ Correct' if voicetree['correct'] else '❌ Incorrect'}")
            print(f"  Baseline Answer: {baseline['answer'][:100]}...")
            print(f"  VoiceTree Answer: {voicetree['answer'][:100]}...")
        
        print(f"\n=== FINAL SUMMARY ===")
        summary = results['summary']
        print(f"📊 Total Questions: {summary['total_questions']}")
        print(f"📈 Baseline Accuracy: {summary['baseline_accuracy']:.1%}")
        print(f"🌳 VoiceTree Accuracy: {summary['voicetree_accuracy']:.1%}")
        
        improvement = summary['improvement']
        if improvement > 0:
            print(f"🎉 VoiceTree Improvement: +{improvement:.1%}")
        elif improvement < 0:
            print(f"📉 VoiceTree Performance: {improvement:.1%}")
        else:
            print(f"🤝 No difference in performance")
        
        print(f"\n📁 Full results saved to: results/{output_file}")
        
    except Exception as e:
        print(f"\n❌ Error running experiment: {e}")
        import traceback
        traceback.print_exc()

def main():
    """Main entry point"""
    print("🌳 VoiceTree Experiment Runner")
    print("=" * 50)
    
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        print("Running test mode...")
        try:
            from test_minimal_benchmark import test_benchmark
            test_benchmark()
        except ImportError:
            print("Test module not found")
    else:
        run_experiment()

if __name__ == "__main__":
    main()
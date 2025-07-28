#!/usr/bin/env python3
"""
Generate the hardest questions for VoiceTree benchmark
Based on paper findings: reverse problems, long context, high operations
"""

import os
import subprocess
import sys

def generate_hard_questions():
    """Generate the hardest questions where LLMs struggle most"""
    
    print("🚀 Generating Hardest Questions for VoiceTree Benchmark")
    print("=" * 60)
    print("Based on paper findings:")
    print("- Reverse problems (forwardreverse mode)")  
    print("- Long context (32k length)")
    print("- High operations (20, 19)")
    print("- Hard difficulty (d=3)")
    print()
    
    # Change to data generation directory
    os.chdir('data/realistic')
    
    # Parameters for hardest questions
    hard_configs = [
        {
            'name': 'Hardest Reverse + Long Context + High Ops',
            'target_length': '32k',
            'd': 3,  # hard difficulty
            'operations': [20, 19],  # highest complexity
            'opmax': 30,
            'total': 10  # Generate 10 questions to have variety
        },
        {
            'name': 'Hard Reverse + Medium-Long Context + High Ops', 
            'target_length': '16k',
            'd': 3,  # hard difficulty
            'operations': [18, 17],  # high complexity
            'opmax': 25,
            'total': 10
        },
        {
            'name': 'Hard Reverse + Standard Context + Medium-High Ops',
            'target_length': '8k', 
            'd': 3,  # hard difficulty
            'operations': [15, 14],  # medium-high complexity
            'opmax': 20,
            'total': 10
        }
    ]
    
    print("📋 Generation Plan:")
    for i, config in enumerate(hard_configs):
        print(f"  {i+1}. {config['name']}")
        print(f"     Length: {config['target_length']}, Ops: {config['operations']}, Difficulty: {'hard' if config['d']==3 else 'medium'}")
    print()
    
    # Generate each configuration
    for config in hard_configs:
        print(f"🔄 Generating: {config['name']}")
        
        # Build command
        cmd = [
            'python', 'datagenerationworker.py',
            '--numprocs', '8',
            '--opmax', str(config['opmax']),
            '--total', str(config['total']),
            '--mod', '-1',
            '--number_range', '5',
            '--target_length', config['target_length'],
            '--d', str(config['d']),
            '--force',
            '--listoperations'
        ] + [str(op) for op in config['operations']]
        
        print(f"  Command: {' '.join(cmd)}")
        
        try:
            # Run generation
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            
            if result.returncode == 0:
                print(f"  ✅ Success: {config['name']}")
            else:
                print(f"  ❌ Failed: {config['name']}")
                print(f"  Error: {result.stderr}")
                
        except subprocess.TimeoutExpired:
            print(f"  ⏰ Timeout: {config['name']} (taking longer than 5 minutes)")
        except Exception as e:
            print(f"  ❌ Exception: {e}")
        
        print()
    
    print("🔍 Checking generated files:")
    # Check what was generated
    base_path = "Igsm"
    if os.path.exists(base_path):
        for root, dirs, files in os.walk(base_path):
            for file in files:
                if file.endswith('.jsonl') and 'hard' in root:
                    filepath = os.path.join(root, file)
                    try:
                        with open(filepath, 'r') as f:
                            line_count = sum(1 for line in f)
                        print(f"  📄 {filepath}: {line_count} questions")
                    except:
                        print(f"  📄 {filepath}: Error reading")
    
    print("\n✅ Hard question generation complete!")
    print("Now you can run the Hard VoiceTree Benchmark on these challenging questions.")

if __name__ == "__main__":
    generate_hard_questions()
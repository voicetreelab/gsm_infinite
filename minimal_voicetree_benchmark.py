#!/usr/bin/env python3
"""
Minimal VoiceTree Benchmark for GSM-infinite Dataset

This implements the VoiceTree approach described in VoiceTree_approach.md:
1. Process long context into hierarchical tree structure
2. Let LLM select relevant nodes for answering
3. Compare performance with direct LLM approach
"""

import json
import os
from typing import Dict, List, Tuple, Any
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv('.env')

import sys
sys.path.append('.')
sys.path.append('gsm-infinite')

# Import directly from the pred directory
from pred.model_handler import ModelHandler
from pred.eval_realistic import criteriaoutput


class VoiceTreeProcessor:
    """Processes long contexts into hierarchical tree structures"""
    
    def __init__(self, model_handler: ModelHandler):
        self.model = model_handler
    
    def chunk_context(self, problem_text: str, chunk_size: int = 2000) -> List[str]:
        """Split problem into logical chunks based on sentences"""
        sentences = problem_text.split('. ')
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
    
    def create_node_summary(self, chunk: str) -> str:
        """Create a concise summary of a context chunk"""
        prompt = f"""Summarize this math problem context chunk in 1-2 sentences, focusing on key relationships and constraints:

{chunk}

Summary:"""
        
        return self.model.generate_answer(prompt, max_tokens=150, temperature=0)
    
    def build_tree_view(self, problem_text: str) -> Dict[str, Any]:
        """Build hierarchical tree view of the problem context"""
        chunks = self.chunk_context(problem_text)
        
        tree = {
            "full_context": problem_text,
            "chunks": [],
            "tree_summary": ""
        }
        
        # Create summaries for each chunk
        for i, chunk in enumerate(chunks):
            summary = self.create_node_summary(chunk)
            tree["chunks"].append({
                "id": f"node_{i}",
                "content": chunk,
                "summary": summary
            })
        
        # Create high-level tree summary
        all_summaries = "\n".join([f"Node {i}: {chunk['summary']}" 
                                  for i, chunk in enumerate(tree["chunks"])])
        
        tree_prompt = f"""Create a high-level overview of this problem structure:

{all_summaries}

High-level overview:"""
        
        tree["tree_summary"] = self.model.generate_answer(tree_prompt, max_tokens=200, temperature=0)
        
        return tree
    
    def select_relevant_nodes(self, tree: Dict[str, Any], question: str) -> List[str]:
        """LLM selects which nodes are relevant for answering the question"""
        node_list = "\n".join([f"Node {i}: {chunk['summary']}" 
                              for i, chunk in enumerate(tree["chunks"])])
        
        selection_prompt = f"""Given this question: "{question}"

And these available context nodes:
{node_list}

Which nodes (by number) would be most helpful for answering this question? 
Respond with just the node numbers, comma-separated (e.g., "0,2,4"):"""
        
        response = self.model.generate_answer(selection_prompt, max_tokens=50, temperature=0)
        
        try:
            selected_indices = [int(x.strip()) for x in response.split(',')]
            selected_content = []
            for idx in selected_indices:
                if 0 <= idx < len(tree["chunks"]):
                    selected_content.append(tree["chunks"][idx]["content"])
            return selected_content
        except:
            # Fallback: return first few chunks if parsing fails
            return [chunk["content"] for chunk in tree["chunks"][:2]]


class MinimalVoiceTreeBenchmark:
    """Minimal benchmark comparing Gemini with and without VoiceTree"""
    
    def __init__(self):
        self.gemini_handler = ModelHandler(model_name="gemini-2.5-flash-lite", backend_type="gemini")
        self.voicetree_processor = VoiceTreeProcessor(self.gemini_handler)
        
    def load_sample_questions(self, num_questions: int = 3) -> List[Dict]:
        """Load sample questions from different templates"""
        dataset_path = "data/realistic/Igsm/8k/medium/3/igsm_op3_ip20_force_True_0.jsonl"
        
        questions = []
        templates_seen = set()
        
        with open(dataset_path, 'r') as f:
            for line in f:
                if len(questions) >= num_questions:
                    break
                    
                question = json.loads(line.strip())
                template = question.get('template', 'unknown')
                
                # Try to get variety of templates
                if template not in templates_seen or len(questions) < num_questions:
                    questions.append(question)
                    templates_seen.add(template)
        
        return questions[:num_questions]
    
    def run_baseline_test(self, question: Dict) -> str:
        """Run baseline Gemini test with full context"""
        full_prompt = f"{question['problem']}\n\nQuestion: {question['question']}"
        
        return self.gemini_handler.generate_answer(
            full_prompt, 
            max_tokens=512, 
            temperature=0
        )
    
    def run_voicetree_test(self, question: Dict) -> str:
        """Run VoiceTree-enhanced test"""
        # Step 1: Build tree view
        tree = self.voicetree_processor.build_tree_view(question['problem'])
        
        # Step 2: Select relevant nodes
        selected_nodes = self.voicetree_processor.select_relevant_nodes(
            tree, question['question']
        )
        
        # Step 3: Generate answer with reduced context
        reduced_context = "\n\n".join(selected_nodes)
        prompt = f"{reduced_context}\n\nQuestion: {question['question']}"
        
        return self.gemini_handler.generate_answer(
            prompt,
            max_tokens=512,
            temperature=0
        )
    
    def evaluate_answer(self, generated_answer: str, question: Dict) -> bool:
        """Evaluate if generated answer is correct"""
        # Use existing evaluation logic from eval_realistic.py
        corrected, total = criteriaoutput([generated_answer], question)
        return corrected > 0
    
    def run_benchmark(self) -> Dict[str, Any]:
        """Run the complete minimal benchmark"""
        questions = self.load_sample_questions(3)
        results = {
            "baseline_results": [],
            "voicetree_results": [],
            "summary": {}
        }
        
        print(f"Running benchmark on {len(questions)} questions...")
        
        for i, question in enumerate(questions):
            print(f"\nQuestion {i+1}/{len(questions)} (Template: {question.get('template', 'unknown')})")
            print(f"Question: {question['question']}")
            
            # Baseline test
            print("Running baseline test...")
            baseline_answer = self.run_baseline_test(question)
            baseline_correct = self.evaluate_answer(baseline_answer, question)
            
            results["baseline_results"].append({
                "question_id": question.get('id', i),
                "template": question.get('template', 'unknown'),
                "answer": baseline_answer,
                "correct": baseline_correct
            })
            
            # VoiceTree test
            print("Running VoiceTree test...")
            voicetree_answer = self.run_voicetree_test(question)
            voicetree_correct = self.evaluate_answer(voicetree_answer, question)
            
            results["voicetree_results"].append({
                "question_id": question.get('id', i),
                "template": question.get('template', 'unknown'),
                "answer": voicetree_answer,
                "correct": voicetree_correct
            })
            
            print(f"Baseline: {'✓' if baseline_correct else '✗'}")
            print(f"VoiceTree: {'✓' if voicetree_correct else '✗'}")
        
        # Calculate summary statistics
        baseline_accuracy = sum(r["correct"] for r in results["baseline_results"]) / len(questions)
        voicetree_accuracy = sum(r["correct"] for r in results["voicetree_results"]) / len(questions)
        
        results["summary"] = {
            "total_questions": len(questions),
            "baseline_accuracy": baseline_accuracy,
            "voicetree_accuracy": voicetree_accuracy,
            "improvement": voicetree_accuracy - baseline_accuracy
        }
        
        return results
    
    def save_results(self, results: Dict[str, Any], output_file: str = "minimal_benchmark_results.json"):
        """Save benchmark results to file"""
        os.makedirs("results", exist_ok=True)
        output_path = os.path.join("results", output_file)
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\nResults saved to: {output_path}")


def main():
    """Run the minimal VoiceTree benchmark"""
    print("=== Minimal VoiceTree Benchmark ===")
    print("Comparing Gemini performance with and without VoiceTree context processing")
    
    benchmark = MinimalVoiceTreeBenchmark()
    results = benchmark.run_benchmark()
    
    # Print summary
    print("\n=== RESULTS SUMMARY ===")
    print(f"Total Questions: {results['summary']['total_questions']}")
    print(f"Baseline Accuracy: {results['summary']['baseline_accuracy']:.2%}")
    print(f"VoiceTree Accuracy: {results['summary']['voicetree_accuracy']:.2%}")
    print(f"Improvement: {results['summary']['improvement']:+.2%}")
    
    benchmark.save_results(results)


if __name__ == "__main__":
    main()
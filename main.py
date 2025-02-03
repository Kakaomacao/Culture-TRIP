import argparse
from iterative_refinement.graph_workflow import culture_trip

# Example
# python main.py --prompt "Ao dai are scattered on the floor after a long day." --culture_noun "ao dai"

def main():
    parser = argparse.ArgumentParser(description="Culture-TRIP")
    parser.add_argument("--prompt", type=str, required=True, help="Input prompt for refinement")
    parser.add_argument("--culture_noun", type=str, required=True, help="Culture noun to focus on")
    parser.add_argument("--score_threshold", type=int, required=False, default=40, help="Score threshold")
    parser.add_argument("--is_intermediate_result_show", type=bool, required=False, default=True, help="Culture noun to focus on")
    
    args = parser.parse_args()

    prompt = args.prompt
    culture_noun = args.culture_noun
    score_threshold = args.score_threshold
    is_intermediate_result_show = args.is_intermediate_result_show
    
    refined_prompt = culture_trip(culture_noun, prompt, score_threshold, is_intermediate_result_show)
    print('\n\n### REFINED_PROMPT ###')
    print(refined_prompt)

if __name__ == "__main__":
    main()
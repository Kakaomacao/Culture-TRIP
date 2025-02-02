import argparse
from iterative_refinement.graph_workflow import culture_trip

# Example
# python main.py --prompt "Ao dai are scattered on the floor after a long day." --culture_noun "ao dai"

def main():
    parser = argparse.ArgumentParser(description="Culture-TRIP")
    parser.add_argument("--prompt", type=str, required=True, help="Input prompt for refinement")
    parser.add_argument("--culture_noun", type=str, required=True, help="Culture noun to focus on")
    args = parser.parse_args()

    prompt = args.prompt
    culture_noun = args.culture_noun
    
    refined_prompt = culture_trip(culture_noun, prompt)
    print(refined_prompt)

if __name__ == "__main__":
    main()
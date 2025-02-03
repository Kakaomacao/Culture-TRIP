# Culture-TRIP

This is the official repository for our NAACL 2025 paper **Culture-TRIP: Culturally-Aware Text-to-Image Generation with Iterative Prompt Refinement**

[[Paper]]() | [[Project]](https://shane3606.github.io/Culture-TRIP/)

## Installation

```bash
git clone https://github.com/Kakaomacao/Culture-TRIP.git
conda env create -f environment.yaml
conda activate culture-trip
```

## Usage

### Arguments

| Argument                        | Type | Required | Default | Description                          |
| ------------------------------- | ---- | -------- | ------- | ------------------------------------ |
| `--prompt`                      | str  | Yes      | -       | Input prompt for refinement          |
| `--culture_noun`                | str  | Yes      | -       | Cultural noun to focus on            |
| `--score_threshold`             | int  | No       | 40      | Score threshold                      |
| `--is_intermediate_result_show` | bool | No       | True    | Whether to show intermediate results |

### Example Usage

```bash
python main.py --prompt "Describe a traditional festival" --culture_noun "Diwali" --score_threshold 40 --is_intermediate_result_show True
```

## Citation

Consider citing as below if you find this repository helpful to your project:

```bash

```

## Acknowldement

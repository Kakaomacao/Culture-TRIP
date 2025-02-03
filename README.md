# Culture-TRIP

Culture-TRIP is a Python program that generates refined prompts based on a given input prompt and a cultural noun by itreative prompt refinement.

## Environment Setup

The project environment can be set up using the `environment.yaml` file.

### Create and Activate Conda Environment
```bash
git clone https://github.com/Kakaomacao/Culture-TRIP.git
conda env create -f environment.yaml
conda activate culture-trip
```

## Usage

### Arguments

| Argument                        | Type   | Required | Default | Description                           |
|----------------------------------|--------|----------|---------|---------------------------------------|
| `--prompt`                      | str    | Yes      | -       | Input prompt for refinement          |
| `--culture_noun`                | str    | Yes      | -       | Cultural noun to focus on            |
| `--score_threshold`             | int    | No       | 40      | Score threshold                       |
| `--is_intermediate_result_show` | bool   | No       | True    | Whether to show intermediate results |

### Example Usage

```bash
python main.py --prompt "Describe a traditional festival" --culture_noun "Diwali" --score_threshold 40 --is_intermediate_result_show True
```

## Acknowldement
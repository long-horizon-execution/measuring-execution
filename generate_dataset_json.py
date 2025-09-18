import json
import sys
import os

sys.path.append('src')

from src.experiments.dict_sum.dict_sum_util import DictSumUtil


def generate_string_crud_data():
    """Generate string CRUD prompts and ground truth data."""

    k_operations = 10000
    num_entries = 1000
    seed = 42
    initial_string = ""
    available_chars = "abcdefghijklmnopqrstuvwxyz"

    generator = StringCrudGenerator(
        k_operations=k_operations,
        num_entries=num_entries,
        seed=seed,
        initial_string=initial_string,
        available_chars=available_chars
    )
    
    ground_truth_data = generator.entries

    data = {
        "experiment_type": "string_crud",
        "config": {
            "k_operations": k_operations,
            "num_samples": num_entries,
            "initial_string": initial_string,
            "available_chars": available_chars
        },
        "ground_truth_data": ground_truth_data,
    }
    
    return data


def generate_prefix_sum_data():
    """Generate prefix sum prompts and ground truth data."""

    k = 10000
    num_entries = 1000
    seed = 42
    max_input_value = 9
    min_input_value = -9
    
    generator = PrefixSumGenerator(
        k=k,
        num_entries=num_entries,
        seed=seed,
        max_input_value=max_input_value,
        min_input_value=min_input_value
    )
    ground_truth_data = generator.entries
    
    data = {
        "experiment_type": "prefix_sum",
        "config": {
            "k_digits": k,
            "num_samples": num_entries,
            "max_input_value": max_input_value,
            "min_input_value": min_input_value
        },
        "ground_truth_data": ground_truth_data,
    }
    
    return data


def generate_dict_sum_data(dict_size=100, horizon_length=50000, num_instances=100, 
                          min_input_value=-99, max_input_value=99, seed=42):
    """Generate dict sum prompts and ground truth data."""
    
    generator = DictSumUtil(
        num_pairs=dict_size,
        min_value=min_input_value,
        max_value=max_input_value,
        horizon_length=horizon_length,
        num_instances=num_instances,
        seed=seed
    )
    ground_truth_data = generator.entries
    
    data = {
        "experiment_type": "dict_sum",
        "config": {
            "dict_size": dict_size,
            "horizon_length": horizon_length,
            "num_samples": num_instances,
            "min_input_value": min_input_value,
            "max_input_value": max_input_value
        },
        "dictionary": generator.dict,
        "ground_truth_data": ground_truth_data,
    }
    
    return data
    
# print("Generating string_crud dataset")
# string_crud_data = generate_string_crud_data()

# print("Generating prefix_sum dataset")
# prefix_sum_data = generate_prefix_sum_data()

print("Generating dict_sum dataset")
dict_sum_data = generate_dict_sum_data()

final_dataset = {
    # "string_crud": string_crud_data,
    # "prefix_sum": prefix_sum_data,
    "dict_sum": dict_sum_data
}

output_file = f"dict_sum_{dict_sum_data['config']['dict_size']}.json"
with open(output_file, 'w') as f:
    json.dump(final_dataset, f, indent=2)

print("Datasets generated successfully!")
# print(f"String CRUD: {len(string_crud_data['ground_truth_data'])} samples with {len(string_crud_data['ground_truth_data'][0]['operations'])} operations")
# print(f"Prefix Sum: {len(prefix_sum_data['ground_truth_data'])} samples with {len(prefix_sum_data['ground_truth_data'][0]['input_list'])} digits")
print(f"Dict Sum: {len(dict_sum_data['ground_truth_data'])} samples with {dict_sum_data['config']['dict_size']} keys")

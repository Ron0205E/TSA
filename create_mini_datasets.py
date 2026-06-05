
import csv
import random
import os
from collections import defaultdict


def create_mini_advbench(input_file, output_file, sample_ratio=0.4):
    print("\n" + "=" * 80)
    print("Creating Mini AdvBench Dataset")
    print("=" * 80)

    # Load raw data
    with open(input_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        all_data = list(reader)

    print(f"Original dataset size: {len(all_data)} entries")

    # Calculate sample amount
    sample_size = int(len(all_data) * sample_ratio)
    print(f"Sampling ratio: {sample_ratio * 100:.0f}%")
    print(f"Target sample count: {sample_size} entries")

    # Fix random seed for reproducibility
    random.seed(42)

    # Random sampling
    sampled_data = random.sample(all_data, sample_size)

    # Export to new csv file
    with open(output_file, 'w', encoding='utf-8', newline='') as f:
        fieldnames = ['prompt', 'target']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(sampled_data)

    print(f"✓ Mini AdvBench generated successfully: {len(sampled_data)} entries")
    print(f"✓ Output path: {output_file}")

    return len(sampled_data)


def create_mini_harmbench(input_file, output_file, samples_per_category=20):
    """
    Sample fixed number of entries per category from HarmBench dataset

    Args:
        input_file: Path of source CSV file
        output_file: Path of exported CSV file
        samples_per_category: Number of samples extracted from each semantic category
    """
    print("\n" + "=" * 80)
    print("Creating Mini HarmBench Dataset")
    print("=" * 80)

    # Load raw dataset
    with open(input_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        all_data = list(reader)

    print(f"Original dataset size: {len(all_data)} entries")

    # Group records by semantic category
    categories = defaultdict(list)
    for item in all_data:
        semantic_cat = item.get('SemanticCategory', 'unknown')
        categories[semantic_cat].append(item)

    print(f"\nDetected {len(categories)} semantic categories:")
    for cat, items in sorted(categories.items(), key=lambda x: len(x[1]), reverse=True):
        print(f"  - {cat}: {len(items)} entries")

    # Fix random seed for reproducibility
    random.seed(42)

    # Sample data from every single category
    sampled_data = []
    category_samples = {}

    for cat, items in categories.items():
        sample_size = min(samples_per_category, len(items))
        samples = random.sample(items, sample_size)
        sampled_data.extend(samples)
        category_samples[cat] = sample_size

    # Write sampled result to file
    with open(output_file, 'w', encoding='utf-8', newline='') as f:
        if sampled_data:
            fieldnames = list(sampled_data[0].keys())
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(sampled_data)

    print(f"\n✓ Mini HarmBench generated successfully: {len(sampled_data)} entries")
    print(f"✓ Sampling count per category:")
    for cat, count in sorted(category_samples.items(), key=lambda x: x[1], reverse=True):
        print(f"    - {cat}: {count} entries")
    print(f"✓ Output path: {output_file}")

    return len(sampled_data), category_samples


def main():
    """Main execution entry"""
    # Configure file directory path
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    dataset_dir = os.path.join(project_root, 'dataset')

    # Source input file paths
    advbench_input = os.path.join(dataset_dir, 'advbench.csv')
    harmbench_input = os.path.join(dataset_dir, 'harmbench_behaviors_text_all.csv')

    # Target output file paths
    mini_advbench_output = os.path.join(dataset_dir, 'miniAdvbench.csv')
    mini_harmbench_output = os.path.join(dataset_dir, 'miniHarmbench.csv')

    print("\n" + "=" * 80)
    print("Mini Dataset Generation Tool")
    print("=" * 80)
    print(f"Input Files:")
    print(f"  - AdvBench: {advbench_input}")
    print(f"  - HarmBench: {harmbench_input}")
    print(f"\nOutput Files:")
    print(f"  - Mini AdvBench: {mini_advbench_output}")
    print(f"  - Mini HarmBench: {mini_harmbench_output}")

    # Build Mini AdvBench (40% sampling ratio ~208 items)
    advbench_count = create_mini_advbench(advbench_input, mini_advbench_output, sample_ratio=0.4)

    # Build Mini HarmBench (20 samples for each category)
    harmbench_count, category_samples = create_mini_harmbench(
        harmbench_input,
        mini_harmbench_output,
        samples_per_category=20
    )

    # Final summary log
    print("\n" + "=" * 80)
    print("All datasets created!")
    print("=" * 80)
    print(f"\nGenerated File List:")
    print(f"  - Mini AdvBench: {mini_advbench_output} ({advbench_count} entries)")
    print(f"  - Mini HarmBench: {mini_harmbench_output} ({harmbench_count} entries)")
    print(f"\nTotal entries: {advbench_count + harmbench_count}")
    print()


if __name__ == "__main__":
    main()
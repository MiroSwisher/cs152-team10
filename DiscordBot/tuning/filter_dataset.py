import pandas as pd
from pathlib import Path

def filter_dataset():
    """
    Filter out round 1 examples from v0.2.3 of the dataset.
    Round 1 examples can be identified by:
    1. Having 'round' value of '1' or '1a'
    2. Having 'type' marked as 'notgiven'
    """
    # Read the dataset
    input_path = Path('Dynamically-Generated-Hate-Speech-Dataset-main/Dynamically Generated Hate Dataset v0.2.3.csv')
    df = pd.read_csv(input_path)
    
    # Print initial statistics
    print("\nInitial Dataset Statistics:")
    print(f"Total examples: {len(df)}")
    print("\nDistribution by Round:")
    print(df['round'].value_counts().sort_index())
    print("\nDistribution by Type:")
    print(df['type'].value_counts())
    
    # Filter out round 1 examples
    filtered_df = df[
        ~(
            # Remove examples from round 1
            (df['round'].isin(['1', '1a'])) |
            # Remove examples with type 'notgiven'
            (df['type'] == 'notgiven')
        )
    ]
    
    # Print filtered statistics
    print("\nFiltered Dataset Statistics:")
    print(f"Total examples: {len(filtered_df)}")
    print("\nDistribution by Round:")
    print(filtered_df['round'].value_counts().sort_index())
    print("\nDistribution by Type:")
    print(filtered_df['type'].value_counts())
    
    # Save filtered dataset
    output_path = Path('DiscordBot/hate_speech_dataset_filtered.csv')
    filtered_df.to_csv(output_path, index=False)
    print(f"\nFiltered dataset saved to: {output_path}")

if __name__ == "__main__":
    filter_dataset() 
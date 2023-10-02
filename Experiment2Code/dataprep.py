import os
import pandas as pd

def find_csv_files_in_directory(directory, filename="test_results.csv"):
    """Finds all paths to a specific filename within a directory."""
    matches = []
    for root, dirnames, filenames in os.walk(directory):
        for filename in filenames:
            if filename == "test_results.csv":
                matches.append(os.path.join(root, filename))
    return matches

def main():
    directory_path = './'  # specify your directory path here
    
    csv_files = find_csv_files_in_directory(directory_path)
    
    # List to store individual dataframes
    list_of_dfs = []
    
    for csv_file in csv_files:
        dir_name = os.path.basename(os.path.dirname(csv_file))
        
        # Read the CSV and add a column indicating its originating directory
        df = pd.read_csv(csv_file)
        df['Source_Directory'] = dir_name
        df = df.drop(columns=['Client ID', 'Evaluation Round'])
        
        # Recalculate and rename the 'loss' column to 'accuracy'
        df['accuracy'] = 1 - df['Loss']
        df = df.drop(columns=['Loss'])  # Drop the 'loss' column
        
        list_of_dfs.append(df)
    
    # Concatenate all individual dataframes into a single one
    consolidated_df = pd.concat(list_of_dfs, ignore_index=True)
    
    # Save the consolidated dataframe to a new CSV
    consolidated_df.to_csv('consolidated_results.csv', index=False)

if __name__ == "__main__":
    main()

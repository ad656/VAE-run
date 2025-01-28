import os
import pandas as pd

def read_csv_files_to_dataframe(directory):
    # Initialize an empty DataFrame
    combined_data = pd.DataFrame(columns=["File_Name", "Slice", "Original_Median", "Original_IQR", "Generated_Median", "Generated_IQR"])
    
    # Iterate through all files in the specified directory
    for file_name in os.listdir(directory):
        if file_name.endswith(".csv") and file_name != "test_patients.csv":
            print(file_name)
            file_path = os.path.join(directory, file_name)
            
            # Read the CSV file into a DataFrame
            try:
                data = pd.read_csv(file_path)
                # Filter rows: line has more than 20 characters and starts with 'f_'
                filtered_df = data[data['Original_Median'].notna()]
                print(filtered_df.head())
                
                # Append the filtered data to the combined DataFrame
                combined_data = pd.concat([combined_data, filtered_df], ignore_index=True)
            except Exception as e:
                print(f"Error reading file {file_name}: {e}")
    
    return combined_data

# Directory containing your CSV files
directory_path = "C:/Users/allan/Downloads/ugcsv"

# Read and process the CSV files
result_df = read_csv_files_to_dataframe(directory_path)
print(result_df.head())

column1_sum = result_df['Generated_Median'].sum()
column2_sum = result_df['Original_Median'].sum()
num_rows = len(result_df)



result_median = abs(column1_sum - column2_sum) / num_rows
result_median2 =(column1_sum/column2_sum)
column1_sum = result_df['Generated_IQR'].sum()
column2_sum = result_df['Original_IQR'].sum()

result_iqr = abs(column1_sum - column2_sum) / num_rows
result_iqr2 = abs(column1_sum/column2_sum) 
print(result_median, result_iqr, result_median2, result_iqr2)


# Save the combined DataFrame to a new CSV file
#output_path = "filtered_combined_data.csv"
#result_df.to_csv(output_path, index=False)



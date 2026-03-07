import os
import shutil


def copy_data_to_myresult(target_path):
    # Define the source and destination directories
    current_directory = os.getcwd()
    data_directory = os.path.join(current_directory, 'data')
    result_directory = target_path#os.path.join(target_path, 'result')

    # Remove the existing result directory if it exists
    #if os.path.exists(result_directory):
        #shutil.rmtree(result_directory)
        #print(f"Removed existing result directory: {result_directory}")

    # Create the result directory
    #os.makedirs(result_directory)
    #print(f"Created new result directory: {result_directory}")

    # Copy all items from the data directory to the result directory
    for item in os.listdir(data_directory):
        source_path = os.path.join(data_directory, item)
        destination_path = os.path.join(result_directory, item)

        try:
            # Copy the file or directory
            if os.path.isdir(source_path):
                shutil.copytree(source_path, destination_path)
            else:
                shutil.copy2(source_path, destination_path)
            #print(f"Copied {source_path} to {destination_path}")
        except Exception as e:
            print(f"Error copying {source_path}: {e}")


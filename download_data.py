import os
import urllib.request
import zipfile

def download_and_extract():
    url = "https://archive.ics.uci.edu/static/public/275/bike+sharing+dataset.zip"
    zip_path = "bike_sharing_dataset.zip"
    data_dir = "data"
    
    # Create data directory if it doesn't exist
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        print(f"Created directory: {data_dir}")
        
    print(f"Downloading dataset from {url}...")
    try:
        urllib.request.urlretrieve(url, zip_path)
        print("Download complete.")
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        print("Please download the zip file manually from the link above, extract 'hour.csv', and place it in the 'TimeSeriesForecasting/data/' folder.")
        return
        
    print(f"Extracting hour.csv to {data_dir}...")
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            # Extract only hour.csv
            zip_ref.extract("hour.csv", data_dir)
        print("Extraction complete.")
    except Exception as e:
        print(f"Error extracting zip file: {e}")
    finally:
        if os.path.exists(zip_path):
            os.remove(zip_path)
            print("Cleaned up temporary zip file.")

if __name__ == "__main__":
    # Change working directory to the script's directory to ensure paths are relative to it
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    download_and_extract()

import os
from zipfile import ZipFile
from pathlib import Path
import warnings
warnings.simplefilter("ignore")

def fetch_datasets_from_kaggle(raw_dir):
    """
    TASK:
    1. Download dataset from Kaggle.
    2. Unzip and load to /data/raw folder.
    3. Delete the zip file for clean workspace.
    """

    # Map each stream to its Kaggle repository slug
    stream_manifest = {
        "airport-turnaround": "shazam0k/airport-turnaround-dataset",
        "ppe-compliance": "shazam0k/airport-ppe-dataset",
        "fod-data": "kilogrand/foreign-object-debris-in-airports-fod-a-dataset"
    }

    for stream_name, repository_slug in stream_manifest.items():
        destination_path = raw_dir/stream_name
        destination_path.mkdir(parents=True, exist_ok=True)

        print(f"\n📡 Downloading raw data for {stream_name}...")
        os.system(f"kaggle datasets download -d {repository_slug} --path {destination_path}")

        zip_archives = list(destination_path.glob("*.zip"))
        if zip_archives:
            print(f"📦 Unpacking compressed data into {stream_name}")
            with ZipFile(zip_archives[0], 'r') as archive_ref:
                archive_ref.extractall(destination_path)

            # Instantly delete zip file for clean workspace
            zip_archives[0].unlink()
            print(f"🧹 Deleted the zip file of {stream_name} for clean workspace!")
        
    print(f"\n✅ Successfully extracted Kaggle Datasets.")

        

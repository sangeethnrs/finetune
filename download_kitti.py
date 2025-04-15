import os
import urllib.request
import zipfile

# KITTI URLs
KITTI_URLS = {
    "velodyne": "https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_velodyne.zip",
    "calib": "https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_calib.zip",
    "labels": "https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_label_2.zip"
}

# Output paths
BASE_DIR = os.path.join(os.getcwd(), "data", "kitti")
ZIP_DIR = os.path.join(BASE_DIR, "zips")

def download_file(url, output_path):
    if os.path.exists(output_path):
        print(f"[SKIP] {output_path} already exists.")
        return
    print(f"[DOWNLOADING] {url}")
    urllib.request.urlretrieve(url, output_path)
    print(f"[SAVED] {output_path}")

def unzip_file(zip_path, extract_to):
    print(f"[UNZIPPING] {zip_path}")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    print(f"[EXTRACTED] {extract_to}")

def organize_kitti_folders():
    os.makedirs(ZIP_DIR, exist_ok=True)

    for key, url in KITTI_URLS.items():
        zip_path = os.path.join(ZIP_DIR, f"{key}.zip")
        download_file(url, zip_path)
        unzip_file(zip_path, ZIP_DIR)

        raw_folder = os.path.join(ZIP_DIR, f"training", key if key != "labels" else "label_2")
        target_folder = os.path.join(BASE_DIR, {
            "velodyne": "velodyne",
            "calib": "calib",
            "labels": "labels"
        }[key])

        os.makedirs(target_folder, exist_ok=True)

        # Move the files
        for file_name in os.listdir(raw_folder):
            src = os.path.join(raw_folder, file_name)
            dst = os.path.join(target_folder, file_name)
            os.rename(src, dst)

    print("\n✅ KITTI data is ready in `data/kitti`.")

if __name__ == "__main__":
    organize_kitti_folders()


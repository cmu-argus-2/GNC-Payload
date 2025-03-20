import os
import requests
from itertools import product

from requests.exceptions import ChunkedEncodingError
from tqdm import tqdm


OUTPUT_DIR = os.path.join(__file__, "../blue_marble")
BASE_URL = "https://eoimages.gsfc.nasa.gov/images/imagerecords"
DATASET_IDS_BY_MONTH = ["73938", "73967", "73992", "74017", "74042", "76487", "74092", "74117", "74142", "74167", "74192", "74218"]
MONTH_NAMES = ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"]
RESOLUTION = "3x21600x21600"


def download_image(url: str, output_path: str, max_retries=100) -> None:
    """
    Download an image from a URL to a file.
    If the download fails, retry up to max_retries times.
    Each retry will resume the download from where it left off.

    :param url: The URL of the image to download.
    :param output_path: The path to save the downloaded image to.
    :param max_retries: The maximum number of times to retry the download.
    """
    # get file metadata
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3",
        "Accept-Encoding": "identity",
        "Connection": "keep-alive",
    }
    response = requests.head(url, headers=headers)
    response.raise_for_status()
    total_length = response.headers.get("content-length")
    assert total_length is not None
    assert response.headers.get("Content-Type") == "image/png"
    assert response.headers.get("Accept-Ranges") == "bytes"

    with open(output_path, "wb") as f, tqdm(total=int(total_length), unit="B", unit_scale=True,
                                            unit_divisor=1024, desc="Downloading") as pbar:
        for retry in range(max_retries):
            try:
                headers["Range"] = f"bytes={pbar.n}-"
                response = requests.get(url, headers=headers, stream=True, timeout=100)
                response.raise_for_status()
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
                    pbar.update(len(chunk))
                print("Download complete")
                return
            except ChunkedEncodingError:
                print(f"Connection closed, resuming download from byte {pbar.n}")
                continue


def main():
    """
    Script entry point.
    """
    for i, (month_name, dataset_id) in enumerate(zip(MONTH_NAMES, DATASET_IDS_BY_MONTH)):
        month_dir = os.path.join(OUTPUT_DIR, month_name)
        os.makedirs(month_dir, exist_ok=True)
        dataset_section_id = dataset_id[:2] + "000"
        for letter, number in product("ABCD", range(1, 3)):
            url = f"{BASE_URL}/{dataset_section_id}/{dataset_id}/world.2004{(i + 1):02d}.{RESOLUTION}.{letter}{number}.png"
            output_path = os.path.join(month_dir, f"{letter}{number}.png")
            print(f"Downloading: {url}")
            download_image(url, output_path)


if __name__ == "__main__":
    main()

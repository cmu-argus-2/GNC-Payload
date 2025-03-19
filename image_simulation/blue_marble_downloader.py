import os
import requests
from itertools import product


OUTPUT_DIR = os.path.join(__file__, "../blue_marble")
BASE_URL = "https://eoimages.gsfc.nasa.gov/images/imagerecords/74000"
DATASET_IDS_BY_MONTH = ["73938", "73967", "73992", "74017", "74042", "76487", "74092", "74117", "74142", "74167", "74192", "74218"]
MONTH_NAMES = ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"]
RESOLUTION = "3x21600x21600"


def main():
    for i, (month_name, dataset_id) in enumerate(zip(MONTH_NAMES, DATASET_IDS_BY_MONTH)):
        month_dir = os.path.join(OUTPUT_DIR, month_name)
        os.makedirs(month_dir, exist_ok=True)
        for letter, number in product("ABCD", range(1, 3)):
            url = f"{BASE_URL}/{dataset_id}/world.2004{(i + 1):02d}.{RESOLUTION}.{letter}{number}.png"
            response = requests.get(url, stream=True)
            with open(os.path.join(month_dir, f"{letter}{number}.png"), "wb") as file:
                file.write(response.content)
        return


if __name__ == "__main__":
    main()

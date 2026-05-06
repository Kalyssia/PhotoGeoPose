import argparse
import json, os, requests
import mercantile
import asyncio
import aiohttp
from collections import defaultdict
from vt2geojson.tools import vt_bytes_to_geojson


# Image directory
SHARED_DIR = "/scratch/users/agraillet"
OUTPUT_DIR = os.path.join(SHARED_DIR, "images")
# Bounding box of Brussels center
BRUSSELS_CENTER = {
    "north": 50.86166,
    "south": 50.83196,
    "west": 4.32501,
    "east": 4.37582
}
# Bounding box of Liege center
LIEGE_CENTER = {
    "north": 50.655012,
    "south": 50.615254,
    "west": 5.555222,
    "east": 5.600235
}
# Default city
CITY_BBOX = BRUSSELS_CENTER
# Zoom for tile retrieval
ZOOM = 14

# City-specific density parameters
DENSITY_PARAMS = {
    "brussels": {
        "cell_size": 0.0001,      # ~10m cells
        "max_per_cell": 12,       # max images per cell
        "max_per_sequence": 10,   # max per sequence
    },
    "liege": {
        "cell_size": 0.00001,     # ~5m cells (smaller = denser grid)
        "max_per_cell": 50,       # more images per cell
        "max_per_sequence": 30,   # more per sequence
    }
}
# Maximum concurrent downloads
MAX_CONCURRENT = 20
# Batch size for URL retrieval
URL_BATCH_SIZE = 50
# Size can be: 256, 1024, or 2048
IMAGE_SIZE: str = "1024"
# Verify access token
ACCESS_TOKEN = os.getenv("MLY_TOKEN")
if not ACCESS_TOKEN:
    raise ValueError("Mapillary access token not found. Please set MLY_TOKEN in your environment.")


# Functions
def angle_diff(a, b) -> float:
    """ Compute the difference between two angles """
    d = abs(a - b) % 360
    return min(d, 360 - d)

def is_angle_diverse(existing, new_angle, angle_threshold) -> bool:
    """ Check if a new angle is different enough compared to existing angles """
    return all(angle_diff(a, new_angle) > angle_threshold for a in existing)

def fetch_tiles() -> list[dict]:
    """ Download metadata from Mapillary tiles """
    # Get the tiles covering the bbox
    west = CITY_BBOX["west"]
    south = CITY_BBOX["south"]
    east = CITY_BBOX["east"]
    north = CITY_BBOX["north"]
    # What impact has the zoom level ?
    tiles = list(mercantile.tiles(west, south, east, north, zooms=ZOOM))
    total_tiles = len(tiles)
    features = []
    for idx, tile in enumerate(tiles, start=1):
        url = f"https://tiles.mapillary.com/maps/vtp/mly1_computed_public/2/{tile.z}/{tile.x}/{tile.y}?access_token={ACCESS_TOKEN}"
        r = requests.get(url)
        try:
            r.raise_for_status()
        except requests.exceptions.HTTPError as e:
            print(f"Failed to fetch tile {tile}: {e}", flush=True)
            continue
        geojson = vt_bytes_to_geojson(r.content, tile.x, tile.y, tile.z, layer="image")
        features.extend(geojson["features"])
        if idx % max((total_tiles // 10), 1) == 0 or idx == total_tiles:
            print(f"Fetched {idx}/{total_tiles} tiles", flush=True)
    # Remove duplicates
    features = list({f["properties"]["id"]: f for f in features if f["geometry"]["type"] == "Point"}.values())
    
    return features

def filter_images(features, city="brussels") -> list[dict]:
    """ Filter images based on spatial, sequence, and angle diversity criteria """
    # Get city-specific density parameters
    params = DENSITY_PARAMS.get(city, DENSITY_PARAMS["brussels"])
    cell_size = params["cell_size"]
    max_per_cell = params["max_per_cell"]
    max_per_sequence = params["max_per_sequence"]
    angle_threshold = 360 // max_per_cell
    
    # Used to filter images based on their location and angle
    grid = defaultdict(list)
    # Used to avoid sequence overrepresentation
    sequence_counts = defaultdict(int)
    selected = []
    for f in features:
        # Only consider points
        if f["geometry"]["type"] != "Point":
            continue
        # Extract coordinates and properties
        lon, lat = f["geometry"]["coordinates"]
        # Ensure the image is in the bbox
        if not (CITY_BBOX["west"] <= lon <= CITY_BBOX["east"] and CITY_BBOX["south"] <= lat <= CITY_BBOX["north"]):
            continue
        properties = f["properties"]
        # We might use it by splitting it in 4 images, but safer to ignore now
        if properties.get("is_pano"):
            continue
        image_id = properties["id"]
        angle = properties.get("compass_angle", 0)
        seq_id = properties.get("sequence_id")
        # If the sequence is overrepresented, skip
        if sequence_counts[seq_id] >= max_per_sequence:
            continue
        # Find the corresponding cell
        cell = (round(lat / cell_size), round(lon / cell_size))
        # If the cell is full, skip
        if len(grid[cell]) >= max_per_cell:
            continue
        # If the angle is not diverse enough, skip
        existing_angles = [x["angle"] for x in grid[cell]]
        if not is_angle_diverse(existing_angles, angle, angle_threshold):
            continue
        # Keep only useful info
        item = {
            "id": image_id,
            "lat": lat,
            "lon": lon,
            "angle": angle,
            "sequence_id": seq_id,
        }
        # Update data structures
        grid[cell].append(item)
        sequence_counts[seq_id] += 1
        selected.append(item)
    # Save it
    json.dump(selected, open(os.path.join(OUTPUT_DIR, "metadata.json"), "w"), indent=2)
    return selected

def fetch_urls(image_ids) -> dict[str, str]:
    """ Fetch image URLs from Mapillary API in batches """
    urls = {}
    total_batches = (len(image_ids) + URL_BATCH_SIZE - 1) // URL_BATCH_SIZE
    for batch_idx, i in enumerate(range(0, len(image_ids), URL_BATCH_SIZE), start=1):
        batch = image_ids[i:i+URL_BATCH_SIZE]
        ids_string = ",".join(map(str, batch))
        url = f"https://graph.mapillary.com?ids={ids_string}&fields=thumb_{IMAGE_SIZE}_url&access_token={ACCESS_TOKEN}"
        # Request the URLs for the batch
        r = requests.get(url)
        try:
            r.raise_for_status()
        except requests.exceptions.HTTPError as e:
            print(f"Failed to fetch URLs for batch {i}: {e}", flush=True)
            continue
        data = r.json()
        # Store the URL for each image
        for k, v in data.items():
            urls[k] = v.get(f"thumb_{IMAGE_SIZE}_url")
        if batch_idx % max((total_batches // 10), 1) == 0 or batch_idx == total_batches:
            print(f"Fetched URL batches {batch_idx}/{total_batches}", flush=True)
    return urls

async def download_image(session, sem, item, url, progress, total):
    """ Download a single image with concurrency control """
    async with sem:
        try:
            async with session.get(url) as resp:
                if resp.status == 200:
                    content = await resp.read()
                    with open(f"{OUTPUT_DIR}/{item['id']}.jpg", "wb") as f:
                        f.write(content)
                    progress["done"] += 1
                    # Print progress every 250 images
                    if progress["done"] % max((total//20), 1) == 0 or progress["done"] == total:
                        print(f"Downloaded {progress['done']}/{total} images", flush=True)
        except Exception:
            print(f"Failed to download image {item['id']} from {url}", flush=True)

async def download_all(items, urls):
    """ Download all images concurrently """
    # Ensure output folder exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    # Limit the number of concurrent downloads
    sem = asyncio.Semaphore(MAX_CONCURRENT)
    # Keep only items with a valid URL so progress has a meaningful total
    downloadable_items = []
    for item in items:
        url = urls.get(str(item["id"]))
        if url:
            downloadable_items.append((item, url))
    total = len(downloadable_items)
    progress = {"done": 0}
    async with aiohttp.ClientSession() as session:
        tasks = [
            download_image(session, sem, item, url, progress, total)
            for item, url in downloadable_items
        ]
        await asyncio.gather(*tasks)
    print(f"Finished: {progress['done']}/{total} images downloaded", flush=True)


# Main function
def parse_args():
    parser = argparse.ArgumentParser(description="Download Mapillary images for a city")
    parser.add_argument(
        "--city",
        choices=["brussels", "liege"],
        default="brussels",
        help="City to download images from (default: brussels)"
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory for images (default: /scratch/users/agraillet/images or ./images if not set)"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    
    global CITY_BBOX, OUTPUT_DIR
    
    # Set city bounding box
    if args.city == "brussels":
        CITY_BBOX = BRUSSELS_CENTER
        print("Using Brussels coordinates")
    elif args.city == "liege":
        CITY_BBOX = LIEGE_CENTER
        print("Using Liege coordinates")
    
    # Set output directory
    if args.output_dir:
        OUTPUT_DIR = args.output_dir
    elif not os.path.exists(SHARED_DIR):
        # Fallback to local directory if shared directory doesn't exist
        OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "images")
        OUTPUT_DIR = os.path.abspath(OUTPUT_DIR)
    
    print(f"Output directory: {OUTPUT_DIR}")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    features = fetch_tiles()
    selected = filter_images(features, city=args.city)
    print(f"Selected {len(selected)} images out of {len(features)} candidates", flush=True)
    ids = [x["id"] for x in selected]
    urls = fetch_urls(ids)
    print("Starting downloads...", flush=True)
    asyncio.run(download_all(selected, urls))
    print("Done.", flush=True)

if __name__ == "__main__":
    main()
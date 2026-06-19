import asyncio, aiohttp, json, mercantile, os, random, requests
from argparse import ArgumentParser
from collections import defaultdict
from dotenv import load_dotenv
from utils import angle_diff_deg, save_json
from vt2geojson.tools import vt_bytes_to_geojson

# Load Mapillary access token from environment variable
load_dotenv()
ACCESS_TOKEN = os.getenv("MLY_TOKEN")   # Token for Mapillary API access
TILE_ZOOM = 14              # Zoom level for tile retrieval
CELL_SIZE = 0.0001          # Cell size in degrees for spatial indexing (~10m²)
MAX_PER_CELL = 12           # Maximum number of images per cell
MAX_PER_SEQ = 10            # Maximum number of images from the same sequence
MAX_CONCURRENT_DL = 20      # Maximum number of concurrent downloads
URL_BATCH_SIZE = 50         # Number of URLs to fetch in one batch from the API
IMAGE_SIZE: str = "1024"    # Image size to download
VAL_RATIO = 0.2             # Ratio of validation data in the final split
SEED = 42                   # Random seed for reproducibility

class DatasetBuilder:
    def __init__(self, output_dir: str, city_file: str, limit_density: bool = True):
        ## Variables
        self.cur_bb: dict[str, float] | None = None
        self.cur_city: str | None = None
        self.cur_dir: str | None = None

        ## Program arguments
        self.output_dir: str = output_dir
        # Ensure output folder exists
        os.makedirs(self.output_dir, exist_ok=True)
        # Load city bounding boxes from configuration file
        with open(city_file, "r", encoding="utf-8") as f:
            # Load all data from the file
            data: dict[str, list[dict]] = json.load(f)
        # Extract the cities from the continents
        cities_data = defaultdict()
        for continent, cities in data.items():
            for city in cities:
                cities_data[city["city"]] = city["coords"]
        # Extract each city's bounding box and store it in a dictionary
        for city_name, city_bb in cities_data.items():
            if not all(k in city_bb for k in ["west", "south", "east", "north"]):
                raise ValueError(f"City '{city_name}' is missing required bounding box keys.")
        self.cities: dict[str, dict[str, float]] = cities_data
        self.limit_density: bool = limit_density

        ## Constants
        self.tile_zoom: int = TILE_ZOOM
        self.cell_size: float = CELL_SIZE
        self.max_per_cell: int = MAX_PER_CELL
        self.min_angle_dif: int = 360 // (2*MAX_PER_CELL)
        self.max_per_seq: int = MAX_PER_SEQ
        self.max_concurrent_dl: int = MAX_CONCURRENT_DL
        self.url_batch_size: int = URL_BATCH_SIZE
        self.image_size: str = IMAGE_SIZE
        self.val_ratio: float = VAL_RATIO
        self.seed: int = SEED
        self.access_token: str = ACCESS_TOKEN
        if not self.access_token:
            raise ValueError("Mapillary access token not found. Please set the MLY_TOKEN environment variable.")

    ## Public Methods

    def build_all(self) -> None:
        """ Filter, download, and split images and metadata for all cities in the configuration file. """
        for city_name in self.cities.keys():
            self.build_city(city_name)

    def build_city(self, city_name: str) -> None:
        """ Filter, download, and split images and metadata for a specific city. """
        if city_name not in self.cities:
            raise ValueError(f"City '{city_name}' not found in configuration. Please choose a valid city or 'all'.")
        self.cur_city = city_name
        self.cur_bb = self.cities[city_name]
        # Create city subdirectory
        self.cur_dir = os.path.join(self.output_dir, city_name)
        os.makedirs(self.cur_dir, exist_ok=True)
        # Load metadata
        features = self._fetch_tiles()
        data = self._filter_images(features)
        print(f"Selected {len(data)} images out of {len(features)} candidates for {city_name}", flush=True)
        ids = [x["id"] for x in data]
        # Get URLs
        urls = self._fetch_urls(ids)
        print(f"Starting downloads for {city_name}...", flush=True)
        # Download images
        asyncio.run(self._download_selection(data, urls))
        # Group metadata by sequence to avoid bias
        grouped = self._group_by_sequence(data)
        sequence_ids = list(grouped.keys())
        train_data, val_data, split_stats = self._split_data(sequence_ids, grouped)
        # Save the splits and stats
        os.makedirs(self.cur_dir, exist_ok=True)
        save_json(train_data, os.path.join(self.cur_dir, "metadata_train.json"))
        save_json(val_data, os.path.join(self.cur_dir, "metadata_val.json"))
        save_json(split_stats, os.path.join(self.cur_dir, "split_stats.json"))


    ## Private Methods

    def _fetch_tiles(self) -> list[dict]:
        """ Download metadata from Mapillary tiles for the current city bounding box. """
        # Get the tiles covering the bbox
        west = self.cur_bb["west"]
        south = self.cur_bb["south"]
        east = self.cur_bb["east"]
        north = self.cur_bb["north"]
        tiles = list(mercantile.tiles(west, south, east, north, zooms=self.tile_zoom))
        total_tiles = len(tiles)
        features = []
        for idx, tile in enumerate(tiles, start=1):
            url = f"https://tiles.mapillary.com/maps/vtp/mly1_computed_public/2/{tile.z}/{tile.x}/{tile.y}?access_token={self.access_token}"
            r = requests.get(url)
            try:
                r.raise_for_status()
            except requests.exceptions.HTTPError as e:
                print(f"Failed to fetch tile {tile}: {e}", flush=True)
                continue
            geojson = vt_bytes_to_geojson(r.content, tile.x, tile.y, tile.z, layer="image")
            features.extend(geojson["features"])
            # Print progress every 10% of tiles processed
            if idx % max((total_tiles // 10), 1) == 0 or idx == total_tiles:
                print(f"Fetched {idx}/{total_tiles} tiles", flush=True)
        # Remove duplicates
        features = list({f["properties"]["id"]: f for f in features if f["geometry"]["type"] == "Point"}.values())
        return features

    def _filter_images(self, features, city="brussels") -> list[dict]:
        """ Filter images based on spatial, sequence, and angle diversity criteria. """
        # Used to filter images based on their location and angle
        grid = defaultdict(list)
        west = self.cur_bb["west"]
        south = self.cur_bb["south"]
        east = self.cur_bb["east"]
        north = self.cur_bb["north"]
        # Used to avoid sequence overrepresentation
        sequence_counts = defaultdict(int)
        selected: list[dict] = []
        for f in features:
            # Only consider points
            if f["geometry"]["type"] != "Point":
                continue
            # Extract coordinates and properties
            lon, lat = f["geometry"]["coordinates"]
            # Ensure the image is in the bbox
            if not (west <= lon <= east and south <= lat <= north):
                continue
            properties = f["properties"]
            # We might use it by splitting it in 4 images, but safer to ignore now
            if properties.get("is_pano"):
                continue
            image_id = properties["id"]
            angle = properties.get("compass_angle", 0)
            seq_id = properties.get("sequence_id")
            # If the sequence is overrepresented, skip
            if sequence_counts[seq_id] >= self.max_per_seq:
                continue
            # Find the corresponding cell
            cell = (round(lat / self.cell_size), round(lon / self.cell_size))
            if self.limit_density:
                # If the cell is full, skip
                if len(grid[cell]) >= self.max_per_cell:
                    continue
                # If the angle is not different enough, skip
                existing_angles = [x["angle"] for x in grid[cell]]
                if not all(angle_diff_deg(a, angle) > self.min_angle_dif for a in existing_angles):
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
        with open(os.path.join(self.cur_dir, "metadata.json"), "w", encoding="utf-8") as f:
            json.dump(selected, f, indent=2)
        return selected

    def _fetch_urls(self, image_ids) -> dict[str, str]:
        """ Fetch image URLs from Mapillary API in batches. """
        urls = {}
        total_batches = (len(image_ids) + self.url_batch_size - 1) // self.url_batch_size
        for batch_idx, i in enumerate(range(0, len(image_ids), self.url_batch_size), start=1):
            batch = image_ids[i:i+self.url_batch_size]
            ids_string = ",".join(map(str, batch))
            url = f"https://graph.mapillary.com?ids={ids_string}&fields=thumb_{self.image_size}_url&access_token={self.access_token}"
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
                urls[k] = v.get(f"thumb_{self.image_size}_url")
            # Print progress every 25% of total
            if batch_idx % max((total_batches // 4), 1) == 0 or batch_idx == total_batches:
                print(f"Fetched URL batches {batch_idx}/{total_batches}", flush=True)
        return urls
    
    async def _download_selection(self, items, urls) -> None:
        """ Download all images concurrently. """
        # Ensure output folder exists
        os.makedirs(self.cur_dir, exist_ok=True)
        # Create the subdirectory for images
        os.makedirs(os.path.join(self.cur_dir, "images"), exist_ok=True)
        # Limit the number of concurrent downloads
        sem = asyncio.Semaphore(self.max_concurrent_dl)
        # Keep only items with a valid URL so progress has a meaningful total
        downloadable_items = []
        for item in items:
            url = urls.get(str(item["id"]))
            if url:
                downloadable_items.append((item, url))
        total = len(downloadable_items)
        progress = {"done": 0}
        # Start concurrent downloads
        async with aiohttp.ClientSession() as session:
            tasks = [
                self._download_image(session, sem, item, url, progress, total)
                for item, url in downloadable_items
            ]
            await asyncio.gather(*tasks)
        print(f"Finished: {progress['done']}/{total} images downloaded", flush=True)

    async def _download_image(self, session, sem, item, url, progress, total) -> None:
        """ Download a single image with concurrency control. """
        async with sem:
            try:
                async with session.get(url) as resp:
                    if resp.status == 200:
                        content = await resp.read()
                        with open(f"{self.cur_dir}/images/{item['id']}.jpg", "wb") as f:
                            f.write(content)
                        progress["done"] += 1
                        # Print progress every 25% of total
                        if progress["done"] % max((total//4), 1) == 0 or progress["done"] == total:
                            print(f"Downloaded {progress['done']}/{total} images", flush=True)
            except Exception:
                print(f"Failed to download image {item['id']} from {url}", flush=True)

    def _group_by_sequence(self, data: list[dict]) -> dict[str, list[dict]]:
        """ Groups annotations by their sequence_id for sequence-level splitting (and not image level) """
        grouped = defaultdict(list)
        for item in data:
            grouped[str(item["sequence_id"])].append(item)
        return dict(grouped)
    
    def _split_data(self, sequence_ids: list[str], grouped: dict[str, list[dict]]) -> tuple[list[dict], list[dict], dict[str, int]]:
        """ Split data into training and validation sets. """
        if not 0.0 < self.val_ratio < 1.0:
            raise ValueError("validation ratio must be between 0 and 1.")
        # Split the sequence IDs
        rng = random.Random(self.seed)
        rng.shuffle(sequence_ids)
        n_val = max(1, int(len(sequence_ids) * self.val_ratio))
        val_seq_ids = set(sequence_ids[:n_val])
        train_seq_ids = set(sequence_ids[n_val:])
        if len(train_seq_ids) == 0:
            raise ValueError("Train split is empty. Reduce val_ratio.")
        # Build the sets
        train_data = []
        val_data = []
        for seq_id, items in grouped.items():
            if seq_id in train_seq_ids:
                train_data.extend(items)
            elif seq_id in val_seq_ids:
                val_data.extend(items)
            else:
                raise RuntimeError(f"Sequence {seq_id} was not assigned to any split.")
        # Compute stats
        stats = {
            "num_train_images": len(train_data),
            "num_train_sequences": len(train_seq_ids),
            "num_val_images": len(val_data),
            "num_val_sequences": len(val_seq_ids),
        }
        return train_data, val_data, stats
    
def parse_args():
    """ Parse command-line arguments. """
    parser = ArgumentParser(
        description="Build a training and validation dataset from Mapillary."
    )
    parser.add_argument(
        "--city",
        default="all",
        help="'All' or the city to download images from (default: brussels)."
    )
    parser.add_argument(
        "--city-file",
        default="cities.json",
        help="Path to the JSON file containing city bounding boxes (default: cities.json)."
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory to save the output (each city will be saved in a subdirectory with its name)."
    )
    parser.add_argument(
        "--no-limit-density",
        dest="limit_density",
        action="store_false",
        help="Disable limiting the density of selected images (default: enabled).",
        default=True,
    )
    return parser.parse_args()

if __name__ == "__main__":
    args= parse_args()
    builder = DatasetBuilder(
        output_dir=args.output_dir,
        city_file=args.city_file,
        limit_density=args.limit_density,
    )
    if args.city == "all":
        builder.build_all()
    else:
        builder.build_city(args.city)
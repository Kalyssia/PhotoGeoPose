import argparse
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

import config
from utils import angle_diff_deg, haversine_m, load_embeddings_metadata, load_topk_results


def tile_image(img, tile_size):
    """Resize and center-crop image into a fixed tile size."""
    tw, th = tile_size
    w, h = img.size
    scale = max(tw / w, th / h)
    nw, nh = int(round(w * scale)), int(round(h * scale))
    img = img.resize((nw, nh), Image.Resampling.BICUBIC)

    left = max(0, (nw - tw) // 2)
    top = max(0, (nh - th) // 2)
    return img.crop((left, top, left + tw, top + th))


def draw_card(draw, x, y, w, h, text, color=(255, 255, 255)):
    """Draw a filled info bar with text."""
    draw.rectangle((x, y, x + w, y + h), fill=(0, 0, 0, 190))
    draw.text((x + 8, y + 6), text, fill=color)


def render_one_row(
    query_id,
    candidates,
    query_meta,
    db_meta,
    image_dir,
    output_path,
    tile_size=(300, 300),
    threshold_m=50.0,
    angle_threshold_deg=None,
):
    """Render one figure: query + top-k retrieved images with proximity labels."""
    padding = 16
    title_h = 70
    info_h = 54
    cols = 1 + len(candidates)
    tile_w, tile_h = tile_size

    canvas_w = padding + cols * tile_w + (cols - 1) * padding + padding
    canvas_h = title_h + padding + tile_h + info_h + padding
    canvas = Image.new("RGB", (canvas_w, canvas_h), color=(245, 246, 248))
    draw = ImageDraw.Draw(canvas, "RGBA")

    font = ImageFont.load_default()

    q_meta = query_meta.get(query_id)
    if q_meta is None:
        raise KeyError(f"Missing query metadata for id={query_id}")

    title = f"Query {query_id} - Top {len(candidates)} retrieved"
    draw.text((padding, 18), title, fill=(20, 20, 20), font=font)

    def image_path(img_id):
        return Path(image_dir) / f"{img_id}.jpg"

    # Query tile
    x = padding
    y = title_h
    q_img = Image.open(image_path(query_id)).convert("RGB")
    q_img = tile_image(q_img, tile_size)
    canvas.paste(q_img, (x, y))
    draw.rectangle((x, y, x + tile_w, y + tile_h), outline=(40, 90, 220), width=4)
    draw_card(draw, x, y + tile_h, tile_w, info_h, f"QUERY id={query_id}", color=(255, 255, 255))

    # Retrieved tiles
    for i, cand in enumerate(candidates):
        rid = int(cand["id"])
        score = float(cand["score"])
        r_meta = db_meta.get(rid)
        if r_meta is None:
            continue

        dist_m = haversine_m(q_meta["lat"], q_meta["lon"], r_meta["lat"], r_meta["lon"])
        ang_d = angle_diff_deg(q_meta["angle"], r_meta["angle"])

        is_close = dist_m <= threshold_m
        if angle_threshold_deg is not None:
            is_close = is_close and (ang_d <= angle_threshold_deg)

        label = "CLOSE" if is_close else "NOT CLOSE"
        border = (40, 160, 80) if is_close else (210, 60, 60)

        x = padding + (i + 1) * (tile_w + padding)
        r_img = Image.open(image_path(rid)).convert("RGB")
        r_img = tile_image(r_img, tile_size)
        canvas.paste(r_img, (x, y))
        draw.rectangle((x, y, x + tile_w, y + tile_h), outline=border, width=4)

        text = f"id={rid} | s={score:.3f} | d={dist_m:.1f}m"
        if angle_threshold_deg is not None:
            text += f" | da={ang_d:.1f}deg"

        draw_card(draw, x, y + tile_h, tile_w, info_h // 2, text, color=(255, 255, 255))
        draw_card(draw, x, y + tile_h + info_h // 2, tile_w, info_h // 2, label, color=border)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def main():
    parser = argparse.ArgumentParser(
        description="Visualize task1 retrieval: query image + top-k retrieved with close/not-close labels"
    )
    parser.add_argument("--results", default=config.VIS_RESULTS, help="Path to top-k retrieval json")
    parser.add_argument(
        "--embeddings", default=config.VIS_EMBEDDINGS, help="Path to embeddings.pt containing db_meta/query_meta"
    )
    parser.add_argument("--image-dir", default=config.IMAGE_DIR, help="Directory containing <id>.jpg")
    parser.add_argument("--output-dir", default=config.VIS_OUTPUT_DIR, help="Directory where visualization images are saved")
    parser.add_argument("--query-id", type=int, default=config.VIS_QUERY_ID, help="Visualize only one query id")
    parser.add_argument("--max-queries", type=int, default=config.VIS_MAX_QUERIES, help="Maximum number of queries to render")
    parser.add_argument("--threshold-m", type=float, default=config.VIS_THRESHOLD_M, help="Distance threshold in meters for CLOSE")
    parser.add_argument(
        "--angle-threshold-deg",
        type=float,
        default=config.VIS_ANGLE_THRESHOLD_DEG,
        help="Optional angle threshold in degrees; if set, CLOSE requires both distance and angle constraints",
    )
    parser.add_argument("--tile-size", type=int, default=config.VIS_TILE_SIZE, help="Square tile size in pixels")

    args = parser.parse_args()


    # load data
    results = load_topk_results(args.results)
    db_meta, query_meta = load_embeddings_metadata(args.embeddings)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.query_id is not None:
        query_ids = [args.query_id]
    else:
        query_ids = sorted(results.keys())[: args.max_queries]

    rendered = 0
    for qid in query_ids:
        if qid not in results:
            print(f"Skip query {qid}: not in results")
            continue
        candidates = results[qid]
        out_path = output_dir / f"query_{qid}_topk.jpg"
        try:
            render_one_row(
                query_id=qid,
                candidates=candidates,
                query_meta=query_meta,
                db_meta=db_meta,
                image_dir=args.image_dir,
                output_path=out_path,
                tile_size=(args.tile_size, args.tile_size),
                threshold_m=args.threshold_m,
                angle_threshold_deg=args.angle_threshold_deg,
            )
            rendered += 1
        except FileNotFoundError as e:
            print(f"Skip query {qid}: missing image file ({e})")
        except KeyError as e:
            print(f"Skip query {qid}: missing metadata ({e})")

    print(f"Rendered {rendered} visualization(s) in {output_dir}")


if __name__ == "__main__":
    main()

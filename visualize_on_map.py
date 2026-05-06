#!/usr/bin/env python3
"""Convert user_results_summary.yaml to GeoJSON for map visualization.

Usage:
    python visualize_on_map.py --input outputs/user_results_summary.yaml --output results.geojson
    
Then open results.geojson in:
    - https://geojson.io (drag & drop)
    - Google My Maps (import)
    - Any GIS software
"""

import argparse
import json
import yaml
import math
from pathlib import Path


def create_geojson(input_yaml, output_geojson):
    """Convert YAML results to GeoJSON format with directional arrows."""
    with open(input_yaml, 'r') as f:
        data = yaml.safe_load(f)
    
    features = []
    
    for img_id, result in data.get('results', {}).items():
        pos = result.get('position_estimate')
        angle = result.get('estimated_angle')
        
        if not pos or pos.get('lat') is None or pos.get('lon') is None:
            continue
            
        lat, lon = pos['lat'], pos['lon']
        
        # Create point feature
        point_feature = {
            "type": "Feature",
            "geometry": {
                "type": "Point",
                "coordinates": [lon, lat]
            },
            "properties": {
                "image_id": img_id,
                "estimated_angle": angle,
                "consistency_error": result.get('consistency_error'),
                "avg_matches_used": result.get('avg_matches_used'),
            }
        }
        features.append(point_feature)
        
        # Add directional arrow only if angle is available and valid
        if angle is not None and not math.isnan(angle):
            # Calculate arrow endpoint (50m in direction of angle)
            # 1 degree ≈ 111km at equator, less at higher latitudes
            lat_rad = math.radians(lat)
            meters_per_deg_lat = 111132.92 - 559.82 * math.cos(2 * lat_rad) + 1.175 * math.cos(4 * lat_rad)
            meters_per_deg_lon = 111412.84 * math.cos(lat_rad) - 93.5 * math.cos(3 * lat_rad)
            
            # 50m arrow length
            delta_lat = 50 * math.cos(math.radians(angle)) / meters_per_deg_lat
            delta_lon = 50 * math.sin(math.radians(angle)) / meters_per_deg_lon
            
            arrow_end_lat = lat + delta_lat
            arrow_end_lon = lon + delta_lon
            
            line_feature = {
                "type": "Feature",
                "geometry": {
                    "type": "LineString",
                    "coordinates": [[lon, lat], [arrow_end_lon, arrow_end_lat]]
                },
                "properties": {
                    "image_id": f"{img_id}_direction",
                    "angle": angle,
                }
            }
            features.append(line_feature)
    
    geojson = {
        "type": "FeatureCollection",
        "features": features
    }
    
    with open(output_geojson, 'w') as f:
        json.dump(geojson, f, indent=2)
    
    print(f"Created {output_geojson} with {len(features)} features")
    print(f"\nView it online:")
    print(f"  1. Open https://geojson.io")
    print(f"  2. Drag and drop {output_geojson}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize results on map")
    parser.add_argument("--input", default="outputs/user_results_summary.yaml",
                        help="Input YAML file")
    parser.add_argument("--output", default="results.geojson",
                        help="Output GeoJSON file")
    args = parser.parse_args()
    
    create_geojson(args.input, args.output)

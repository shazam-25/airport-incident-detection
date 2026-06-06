from pathlib import Path
import yaml
# import os

# def generate_all_yaml_configs(preprocessed_dir_path, fod_classes):
#     """
#     TASK:
#     Generate YAML config files for all streams.
#     """
#     print("📝 Generating multi-stream YAML configuration files...")

#     preprocessed_dir_path = Path(preprocessed_dir_path)
#     config_dir = preprocessed_dir_path / "config"
#     config_dir.mkdir(parents=True, exist_ok=True)

#     configs = {
#         "turnaround_data.yaml": {
#             "path": os.path.join(preprocessed_dir_path, "airport-turnaround"),
#             "train": "train/images", "val": "val/images", "test": "test/images",
#             "nc": 13,
#             "names": ['aircraft', 'baggage_truck', 'bridge_connected', 'bus', 'catering_truck', 'fuel_truck', 'fueling', 'ground_power', 'person', 'pushback_tractor', 'ramp_loader', 'rolling_stairway', 'stairway']
#         },
#         "ppe_data.yaml": {
#             "path": os.path.join(preprocessed_dir_path, "ppe-compliance"),
#             "train": "train/images", "val": "val/images", "test": "test/images",
#             "nc": 4,
#             "names": ['Ear Protectors', 'Safety Vest', 'Without Ear Protectors', 'Without Safety Vest']
#         },
#         "fod_data.yaml": {
#             "path": os.path.join(preprocessed_dir_path, "fod-data"),
#             "train": "train/images", "val": "val/images", "test": "test/images",
#             "nc": len(fod_classes),
#             "names": list(fod_classes)
#         }
#     }

#     for filename, configuration_map  in configs.items():
#         yaml_path = config_dir/f"{filename}"
#         with open(yaml_path, 'w') as out_f:
#             yaml.dump(configuration_map, out_f, default_flow_style=False)
#         print(f"✅ Created configuration map {filename}")


def generate_yaml_profiles(processed_dir, fod_classes):
    """
    Generates unified config file formats with accurate structural path mappings.
    """
    processed_dir = Path(processed_dir)
    config_dir = processed_dir / "config"
    config_dir.mkdir(parents=True, exist_ok=True)

    configs = {
        "turnaround_data.yaml": {
            "path": str(processed_dir / "airport-turnaround"),
            "train": "train/images", "val": "val/images", "test": "test/images",
            "nc": 13,
            "names": ['aircraft', 'baggage_truck', 'bridge_connected', 'bus', 'catering_truck', 'fuel_truck', 'fueling', 'ground_power', 'person', 'pushback_tractor', 'ramp_loader', 'rolling_stairway', 'stairway']
        },
        "ppe_data.yaml": {
            "path": str(processed_dir / "ppe-compliance"),
            "train": "train/images", "val": "val/images", "test": "test/images",
            "nc": 4,
            "names": ['Ear Protectors', 'Safety Vest', 'Without Ear Protectors', 'Without Safety Vest']
        },
        "fod_data.yaml": {
            "path": str(processed_dir / "fod-data"),
            "train": "train/images", "val": "val/images", "test": "test/images",
            "nc": len(fod_classes),
            "names": list(fod_classes)
        }
    }

    for filename, structure in configs.items():
        with open(config_dir / filename, 'w') as f:
            yaml.dump(structure, f, default_flow_style=False)
    print(f"📝 Master configurations written successfully to: {config_dir}")
"""
SUN RGB-D 15 Scene Category Mapping

Based on the paper "SUN RGB-D: A RGB-D Scene Understanding Benchmark Suite"
(arxiv.org/pdf/1911.00155)

Merges the 19 standard categories down to 15 by combining semantically similar
classes identified through silhouette index analysis (Table S2, Page 18).

Merge pairs:
1. classroom + lecture_theatre → classroom
2. home_office + office → office
3. conference_room + study_space → study_space
4. living_room + rest_space → rest_space
"""

# Final 15 scene categories after merging
SUNRGBD_15_CATEGORIES = [
    'bathroom',
    'bedroom',
    'classroom',           # merged: classroom + lecture_theatre
    'computer_room',
    'corridor',
    'dining_area',
    'dining_room',
    'discussion_area',
    'furniture_store',
    'kitchen',
    'lab',
    'library',
    'office',              # merged: home_office + office
    'rest_space',          # merged: living_room + rest_space
    'study_space',         # merged: conference_room + study_space
]

# Mapping from 19 standard categories to 15 merged categories
MERGE_19_TO_15 = {
    # Keep as-is (11 categories)
    'bathroom': 'bathroom',
    'bedroom': 'bedroom',
    'computer_room': 'computer_room',
    'corridor': 'corridor',
    'dining_area': 'dining_area',
    'dining_room': 'dining_room',
    'discussion_area': 'discussion_area',
    'furniture_store': 'furniture_store',
    'kitchen': 'kitchen',
    'lab': 'lab',
    'library': 'library',

    # Merge pairs (4 pairs → 4 categories)
    'classroom': 'classroom',
    'lecture_theatre': 'classroom',  # merged with classroom

    'home_office': 'office',         # merged with office
    'office': 'office',

    'conference_room': 'study_space',  # merged with study_space
    'study_space': 'study_space',

    'living_room': 'rest_space',     # merged with rest_space
    'rest_space': 'rest_space',
}

# Mapping from 45 raw scene types directly to 15 categories
SCENE_MAPPING_45_TO_15 = {
    # Direct mappings to 15 categories
    'bathroom': 'bathroom',
    'bedroom': 'bedroom',
    'classroom': 'classroom',
    'computer_room': 'computer_room',
    'corridor': 'corridor',
    'dining_area': 'dining_area',
    'dining_room': 'dining_room',
    'discussion_area': 'discussion_area',
    'furniture_store': 'furniture_store',
    'kitchen': 'kitchen',
    'lab': 'lab',
    'library': 'library',

    # Merged categories
    'lecture_theatre': 'classroom',      # merged with classroom
    'home_office': 'office',             # merged with office
    'office': 'office',
    'conference_room': 'study_space',    # merged with study_space
    'study_space': 'study_space',
    'living_room': 'rest_space',         # merged with rest_space
    'rest_space': 'rest_space',

    # Map similar scenes to merged categories
    'office_kitchen': 'kitchen',
    'office_dining': 'dining_area',
    'cafeteria': 'dining_area',

    'study': 'study_space',              # → study_space
    'reception_room': 'rest_space',
    'reception': 'rest_space',
    'lobby': 'rest_space',

    'bookstore': 'furniture_store',
    'printer_room': 'computer_room',
    'mail_room': 'office',               # → office

    'recreation_room': 'rest_space',
    'playroom': 'rest_space',
    'gym': 'rest_space',

    'basement': 'rest_space',
    'storage_room': 'rest_space',

    'dinette': 'dining_room',
    'laundromat': 'rest_space',
    'stairs': 'corridor',

    'home': 'rest_space',                # → rest_space (was living_room)
    'hotel_room': 'bedroom',
    'indoor_balcony': 'rest_space',      # → rest_space (was living_room)

    'exhibition': 'rest_space',
    'coffee_room': 'rest_space',
    'dancing_room': 'rest_space',
    'music_room': 'rest_space',

    'idk': None,  # Unknown - will be filtered out
}

# Reverse: get class index from scene name
SCENE_TO_IDX = {scene: idx for idx, scene in enumerate(SUNRGBD_15_CATEGORIES)}

def map_raw_scene_to_15(raw_scene):
    """
    Map a raw scene type (45 types) to one of 15 merged categories.

    Args:
        raw_scene: Raw scene type from scene.txt file

    Returns:
        Merged category name (one of 15) or None if should be filtered
    """
    return SCENE_MAPPING_45_TO_15.get(raw_scene, None)

def map_19_to_15(scene_19):
    """
    Map a 19-category scene to one of 15 merged categories.

    Args:
        scene_19: Scene category from 19-category set

    Returns:
        Merged category name (one of 15)
    """
    return MERGE_19_TO_15.get(scene_19, None)

def get_class_idx(scene_name):
    """
    Get class index (0-14) for a scene name.

    Args:
        scene_name: Scene category name (one of 15)

    Returns:
        Class index 0-14, or None if invalid
    """
    return SCENE_TO_IDX.get(scene_name, None)

def map_and_filter_scenes(scene_list):
    """
    Map list of raw scenes to 15 categories and filter out unknowns.

    Args:
        scene_list: List of raw scene names

    Returns:
        Tuple of (mapped_scenes, filtered_indices)
        - mapped_scenes: List of mapped scene names
        - filtered_indices: Indices of valid samples (unknown filtered out)
    """
    mapped_scenes = []
    filtered_indices = []

    for idx, raw_scene in enumerate(scene_list):
        mapped = map_raw_scene_to_15(raw_scene)
        if mapped is not None:
            mapped_scenes.append(mapped)
            filtered_indices.append(idx)

    return mapped_scenes, filtered_indices

# Print mapping info
if __name__ == "__main__":
    print("SUN RGB-D 15 Merged Categories:")
    print("=" * 60)
    for idx, cat in enumerate(SUNRGBD_15_CATEGORIES):
        print(f"{idx:2d}. {cat}")

    print(f"\n\nMerge mapping (19 → 15):")
    print("=" * 60)

    from collections import defaultdict
    merged_from = defaultdict(list)

    for src, dst in MERGE_19_TO_15.items():
        merged_from[dst].append(src)

    for final_cat in SUNRGBD_15_CATEGORIES:
        sources = merged_from[final_cat]
        if len(sources) == 1:
            print(f"{final_cat:20s} ← {sources[0]} (kept)")
        else:
            print(f"{final_cat:20s} ← {' + '.join(sources)} (merged)")

    print(f"\n\nMapping from 45 raw scenes to 15 final categories:")
    print("=" * 60)

    category_sources = defaultdict(list)

    for raw, final in SCENE_MAPPING_45_TO_15.items():
        if final is not None:
            category_sources[final].append(raw)

    for final_cat in SUNRGBD_15_CATEGORIES:
        sources = category_sources[final_cat]
        if len(sources) == 1 and sources[0] == final_cat:
            print(f"{final_cat:20s} ← {final_cat} (direct)")
        else:
            print(f"{final_cat:20s} ← {', '.join(sorted(sources))}")

    # Show filtered
    filtered = [k for k, v in SCENE_MAPPING_45_TO_15.items() if v is None]
    print(f"\nFiltered out: {filtered}")

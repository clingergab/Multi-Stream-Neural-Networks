"""
SUN RGB-D 19 Standard Scene Category Mapping

Based on research papers using SUN RGB-D for scene classification.
Maps our 45 raw scene types to the standard 19 categories (all with >80 samples).
"""

# Standard 19 scene categories used in SUN RGB-D literature
SUNRGBD_19_CATEGORIES = [
    'bathroom',
    'bedroom',
    'classroom',
    'computer_room',
    'conference_room',
    'corridor',
    'dining_area',
    'dining_room',
    'discussion_area',
    'furniture_store',
    'home_office',
    'kitchen',
    'lab',
    'lecture_theatre',
    'library',
    'living_room',
    'office',
    'rest_space',
    'study_space',
]

# Mapping from our 45 raw scene types to 19 standard categories
SCENE_MAPPING_45_TO_19 = {
    # Direct mappings (same name)
    'bathroom': 'bathroom',
    'bedroom': 'bedroom',
    'classroom': 'classroom',
    'computer_room': 'computer_room',
    'conference_room': 'conference_room',
    'corridor': 'corridor',
    'dining_area': 'dining_area',
    'dining_room': 'dining_room',
    'discussion_area': 'discussion_area',
    'furniture_store': 'furniture_store',
    'home_office': 'home_office',
    'kitchen': 'kitchen',
    'lab': 'lab',
    'lecture_theatre': 'lecture_theatre',
    'library': 'library',
    'living_room': 'living_room',
    'office': 'office',
    'rest_space': 'rest_space',
    'study_space': 'study_space',

    # Merges - map similar scenes to standard categories
    'office_kitchen': 'kitchen',  # Kitchen variant
    'office_dining': 'dining_area',  # Dining variant
    'cafeteria': 'dining_area',  # Another dining variant

    'study': 'study_space',  # Study variant
    'reception_room': 'rest_space',  # Reception is a rest area
    'reception': 'rest_space',  # Same as above
    'lobby': 'rest_space',  # Lobby is for resting/waiting

    'bookstore': 'furniture_store',  # Store variant
    'printer_room': 'computer_room',  # Computer-related room
    'mail_room': 'office',  # Office-related room

    'recreation_room': 'rest_space',  # Recreation area
    'playroom': 'rest_space',  # Play area
    'gym': 'rest_space',  # Exercise area

    'basement': 'rest_space',  # Storage/rest area
    'storage_room': 'rest_space',  # Storage area

    'dinette': 'dining_room',  # Small dining area
    'laundromat': 'rest_space',  # Service area
    'stairs': 'corridor',  # Passage area

    'home': 'living_room',  # General home area
    'hotel_room': 'bedroom',  # Bedroom variant
    'indoor_balcony': 'living_room',  # Living space variant

    'exhibition': 'rest_space',  # Public space
    'coffee_room': 'rest_space',  # Break room
    'dancing_room': 'rest_space',  # Recreation
    'music_room': 'rest_space',  # Recreation

    'idk': None,  # Unknown - will be filtered out
}

# Reverse: get class index from scene name
SCENE_TO_IDX = {scene: idx for idx, scene in enumerate(SUNRGBD_19_CATEGORIES)}

def map_raw_scene_to_19(raw_scene):
    """
    Map a raw scene type (45 types) to one of 19 standard categories.

    Args:
        raw_scene: Raw scene type from scene.txt file

    Returns:
        Standard category name (one of 19) or None if should be filtered
    """
    return SCENE_MAPPING_45_TO_19.get(raw_scene, None)

def get_class_idx(scene_name):
    """
    Get class index (0-18) for a scene name.

    Args:
        scene_name: Scene category name (one of 19)

    Returns:
        Class index 0-18, or None if invalid
    """
    return SCENE_TO_IDX.get(scene_name, None)

def map_and_filter_scenes(scene_list):
    """
    Map list of raw scenes to 19 categories and filter out unknowns.

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
        mapped = map_raw_scene_to_19(raw_scene)
        if mapped is not None:
            mapped_scenes.append(mapped)
            filtered_indices.append(idx)

    return mapped_scenes, filtered_indices

# Print mapping info
if __name__ == "__main__":
    print("SUN RGB-D 19 Standard Categories:")
    print("=" * 60)
    for idx, cat in enumerate(SUNRGBD_19_CATEGORIES):
        print(f"{idx:2d}. {cat}")

    print(f"\n\nMapping from 45 raw scenes to 19 standard:")
    print("=" * 60)

    from collections import defaultdict
    category_sources = defaultdict(list)

    for raw, standard in SCENE_MAPPING_45_TO_19.items():
        if standard is not None:
            category_sources[standard].append(raw)

    for standard_cat in SUNRGBD_19_CATEGORIES:
        sources = category_sources[standard_cat]
        if len(sources) == 1 and sources[0] == standard_cat:
            print(f"{standard_cat:20s} ← {standard_cat} (direct)")
        else:
            print(f"{standard_cat:20s} ← {', '.join(sources)}")

    # Show filtered
    filtered = [k for k, v in SCENE_MAPPING_45_TO_19.items() if v is None]
    print(f"\nFiltered out: {filtered}")

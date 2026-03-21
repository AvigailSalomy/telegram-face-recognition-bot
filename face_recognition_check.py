#!/usr/bin/env python3
"""
Face Recognition Check Script
Compares face1.png with face2.png and face3.png to determine similarity.
"""

import face_recognition
from pathlib import Path


def load_and_encode(image_path: str):
    """Load an image and get its face encoding."""
    image = face_recognition.load_image_file(image_path)
    encodings = face_recognition.face_encodings(image)
    
    if len(encodings) == 0:
        raise ValueError(f"No face found in {image_path}")
    elif len(encodings) > 1:
        print(f"Warning: Multiple faces found in {image_path}, using the first one.")
    
    return encodings[0]


def main():
    # Define image paths
    face1_path = "faces/face1.png"
    face2_path = "faces/face2.png"
    face3_path = "faces/face3.png"
    
    print("Loading images and extracting face encodings...")
    print(f"  - Loading {face1_path}...")
    encoding1 = load_and_encode(face1_path)
    
    print(f"  - Loading {face2_path}...")
    encoding2 = load_and_encode(face2_path)
    
    print(f"  - Loading {face3_path}...")
    encoding3 = load_and_encode(face3_path)
    
    # Compare face1 to face2
    print("\nComparing faces...")
    distance_1_2 = face_recognition.face_distance([encoding1], encoding2)[0]
    is_same_1_2 = face_recognition.compare_faces([encoding1], encoding2)[0]
    
    print(f"  face1 vs face2:")
    print(f"    - Distance: {distance_1_2:.4f}")
    print(f"    - Same person: {is_same_1_2}")
    
    # Compare face1 to face3
    distance_1_3 = face_recognition.face_distance([encoding1], encoding3)[0]
    is_same_1_3 = face_recognition.compare_faces([encoding1], encoding3)[0]
    
    print(f"  face1 vs face3:")
    print(f"    - Distance: {distance_1_3:.4f}")
    print(f"    - Same person: {is_same_1_3}")
    
    # Determine which face is more similar
    print("\nResult:")
    if distance_1_2 < distance_1_3:
        print(f"  face1 is MORE similar to face2 (distance: {distance_1_2:.4f})")
        print(f"  than to face3 (distance: {distance_1_3:.4f})")
    elif distance_1_3 < distance_1_2:
        print(f"  face1 is MORE similar to face3 (distance: {distance_1_3:.4f})")
        print(f"  than to face2 (distance: {distance_1_2:.4f})")
    else:
        print(f"  face1 is EQUALLY similar to both face2 and face3")
        print(f"  (distance to both: {distance_1_2:.4f})")


if __name__ == "__main__":
    main()

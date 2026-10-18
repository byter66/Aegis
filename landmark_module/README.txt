from landmark import landmark_payload
result = landmark_payload("path/to/image.jpg")   # never raises
Keep the models/ folder next to landmark.py. Models load once on import (a few seconds).
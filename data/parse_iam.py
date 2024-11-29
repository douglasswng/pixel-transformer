import xml.etree.ElementTree as ET
import uuid
import json
from constants import IAM_FOLDER, RAW_COORD_FOLDER

def parse_xml(file_path):
    tree = ET.parse(file_path)
    root = tree.getroot()
    return root

def process_xml_content(root):
    strokes = []
    stroke_set = root.find('StrokeSet')
    if stroke_set is not None:
        for stroke in stroke_set.findall('Stroke'):
            points = []
            for point in stroke.findall('Point'):
                x, y = int(point.get('x')), int(point.get('y'))
                points.append((x, y))
            if points:
                strokes.append(points)
    return strokes

def process_strokes(strokes):
    WORD_GAP_THRESHOLD_RATIO = 0.02

    width = max(x for stroke in strokes for x, _ in stroke) - min(x for stroke in strokes for x, _ in stroke)
    words = []
    word = []
    
    for stroke in strokes:
        stroke_l = min(x for x, _ in stroke)
        
        if word:
            word_r = max(x for stroke in word for x, _ in stroke)
            gap = stroke_l - word_r
            
            if gap > width * WORD_GAP_THRESHOLD_RATIO:
                words.append(word)
                word = []
        
        word.append(stroke)
    
    if word:
        words.append(word)
    
    return words

def process_words(words):
    word_dicts = []
    for word in words:
        word_dict = {
            'word': '<unknown>',
            'tokens': []
        }
        
        word_dict['tokens'].append({'type': 'start'})
        
        for stroke in word:
            for x, y in stroke:
                word_dict['tokens'].append({
                    'type': 'coordinate',
                    'x': x,
                    'y': y
                })
            word_dict['tokens'].append({'type': 'pause'})
        
        word_dict['tokens'].pop()
        word_dict['tokens'].append({'type': 'end'})
        
        word_dicts.append(word_dict)
    
    return word_dicts

def save_words(word_dicts):
    for word_dict in word_dicts:
        word_uuid = str(uuid.uuid4())
        
        filename = f"{word_uuid}.json"
        
        file_path = RAW_COORD_FOLDER / filename
        
        with open(file_path, 'w') as f:
            json.dump(word_dict, f, indent=2)

import matplotlib.pyplot as plt

def visualise(word_dicts):
    for i, word_dict in enumerate(word_dicts):
        # Extract x and y coordinates
        x_coords = []
        y_coords = []
        pause_indices = []
        
        for j, token in enumerate(word_dict['tokens']):
            if token['type'] == 'coordinate':
                x_coords.append(token['x'])
                y_coords.append(token['y'])
            elif token['type'] == 'pause':
                pause_indices.append(len(x_coords) - 1)

        # Create a new figure for each word
        plt.figure(figsize=(20, 10))
        plt.title(f"Word {i+1}")

        # Plot the coordinates
        for j in range(len(x_coords)):
            if j == 0:
                plt.plot(x_coords[j], y_coords[j], 'go', markersize=10)  # Green dot for start
            elif j == len(x_coords) - 1:
                plt.plot(x_coords[j], y_coords[j], 'ro', markersize=10)  # Red dot for end
            else:
                plt.plot(x_coords[j], y_coords[j], 'bo')  # Blue dots for points

        # Plot lines between points, with gaps for pauses
        for j in range(len(x_coords) - 1):
            if j in pause_indices:
                # Draw a dashed line for pauses
                plt.plot([x_coords[j], x_coords[j+1]], [y_coords[j], y_coords[j+1]], 'r--', alpha=0.5)
            else:
                plt.plot([x_coords[j], x_coords[j+1]], [y_coords[j], y_coords[j+1]], 'b-')

        # Add arrows to show direction
        for j in range(0, len(x_coords) - 1, max(1, len(x_coords) // 10)):  # Add arrows every 10% of points
            dx = x_coords[j+1] - x_coords[j]
            dy = y_coords[j+1] - y_coords[j]
            plt.arrow(x_coords[j], y_coords[j], dx, dy, shape='full', lw=0, length_includes_head=True, head_width=5)

        # Invert y-axis to match typical image coordinate system
        plt.gca().invert_yaxis()

        # Set aspect ratio to equal for better visualization
        plt.gca().set_aspect('equal', adjustable='box')

        # Add grid
        plt.grid(True, linestyle='--', alpha=0.7)

        # Add labels
        plt.xlabel('X coordinate')
        plt.ylabel('Y coordinate')

        # Add legend
        plt.plot([], [], 'go', label='Start')
        plt.plot([], [], 'ro', label='End')
        plt.plot([], [], 'bo', label='Points')
        plt.plot([], [], 'b-', label='Stroke')
        plt.plot([], [], 'r--', label='Pause')
        plt.legend()

        # Show the plot
        plt.show()

def clear_folder():
    for json_file in RAW_COORD_FOLDER.iterdir():
        json_file.unlink()

def parse_iam():
    word_dicts = []
    for xml_file in sorted(IAM_FOLDER.rglob('*.xml')):
        root = parse_xml(xml_file)
        strokes = process_xml_content(root)
        words = process_strokes(strokes)
        word_dicts.extend(process_words(words))
    save_words(word_dicts)

if __name__ == '__main__':
    clear_folder()
    parse_iam()

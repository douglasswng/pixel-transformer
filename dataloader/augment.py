from typing import List
import random
from dataloader.word_dataclass import Coordinate, Word, Pause
from constants import IMG_H, IMG_W, RAW_COORD_FOLDER

random.seed(42)

def rescale(coordinates: List[Coordinate], scale_factor: float) -> List[Coordinate]:
    rescaled_coords = []
    for coord in coordinates:
        new_x = coord.x * scale_factor
        new_y = coord.y * scale_factor
        rescaled_coords.append(Coordinate(x=new_x, y=new_y))
    return rescaled_coords

def shear(coordinates: List[Coordinate], shear_factor: float) -> List[Coordinate]:
    sheared_coords = []
    for coord in coordinates:
        new_x = coord.x + coord.y * shear_factor
        sheared_coords.append(Coordinate(x=new_x, y=coord.y))
    return sheared_coords

def translate(coordinates: List[Coordinate], shift_x: float, shift_y: float) -> List[Coordinate]:
    translated_coords = []
    for coord in coordinates:
        new_x = coord.x + shift_x
        new_y = coord.y + shift_y
        translated_coords.append(Coordinate(x=new_x, y=new_y))
    return translated_coords

class WordAugmenter:
    def __init__(self, word: Word):
        self.word = word
        self.normalize_padding = 4
        self.rescale_factor_range = (0.8, 1.2)
        self.shear_factor_range = (-0.3, 0.3)
        self.translate_range = (-4, 4)
        self.subsample_factor_range = (0.4, 0.6)

    def _normalize_coordinates(self, coordinates: List[Coordinate], center: bool = True) -> List[Coordinate]:
        min_x = min(coord.x for coord in coordinates)
        min_y = min(coord.y for coord in coordinates)
        max_x = max(coord.x for coord in coordinates)
        max_y = max(coord.y for coord in coordinates)

        new_min_x = self.normalize_padding
        new_min_y = self.normalize_padding
        new_max_x = IMG_W - self.normalize_padding
        new_max_y = IMG_H - self.normalize_padding

        if min_x == max_x and min_y == max_y:
            new_x = (new_min_x + new_max_x) / 2 if center else random.uniform(new_min_x, new_max_x)
            new_y = (new_min_y + new_max_y) / 2 if center else random.uniform(new_min_y, new_max_y)
            return [Coordinate(x=new_x, y=new_y)] * len(coordinates)
        elif min_x == max_x:
            scale = (new_max_y - new_min_y) / (max_y - min_y)
            offset_x = (new_min_x + new_max_x) / 2 if center else random.uniform(new_min_x, new_max_x)
            offset_y = new_min_y
        elif min_y == max_y:
            scale = (new_max_x - new_min_x) / (max_x - min_x)
            offset_x = new_min_x
            offset_y = (new_min_y + new_max_y) / 2 if center else random.uniform(new_min_y, new_max_y)
        else:
            scale_x = (new_max_x - new_min_x) / (max_x - min_x)
            scale_y = (new_max_y - new_min_y) / (max_y - min_y)
            scale = min(scale_x, scale_y)

            scaled_width = (max_x - min_x) * scale
            scaled_height = (max_y - min_y) * scale

            available_width = new_max_x - new_min_x - scaled_width
            available_height = new_max_y - new_min_y - scaled_height

            if center:
                offset_x = new_min_x + available_width / 2
                offset_y = new_min_y + available_height / 2
            else:
                offset_x = new_min_x + random.uniform(0, available_width)
                offset_y = new_min_y + random.uniform(0, available_height)

        normalized_coords = []
        for coord in coordinates:
            new_x = scale * (coord.x - min_x) + offset_x
            new_y = scale * (coord.y - min_y) + offset_y
            normalized_coords.append(Coordinate(x=new_x, y=new_y))

        return normalized_coords
    
    def _random_rescale(self, coordinates: List[Coordinate]) -> List[Coordinate]:
        rescale_factor = random.uniform(*self.rescale_factor_range)
        return rescale(coordinates, rescale_factor)

    def _random_shear(self, coordinates: List[Coordinate]) -> List[Coordinate]:
        shear_factor = random.uniform(*self.shear_factor_range)
        return shear(coordinates, shear_factor)
    
    def _random_translate(self, coordinates: List[Coordinate]) -> List[Coordinate]:
        shift_x = random.uniform(*self.translate_range)
        shift_y = random.uniform(*self.translate_range)
        return translate(coordinates, shift_x, shift_y)
    
    def _subsample(self):
        subsample_factor = random.uniform(*self.subsample_factor_range)
        segments = []
        segment = []
        
        for token in self.word.tokens:
            if isinstance(token, Coordinate):
                segment.append(token)
            else:
                if segment:
                    step = max(1, int(1 / subsample_factor))
                    subsampled_segment = segment[::step]
                    segments.extend(subsampled_segment)
                segments.append(token)
                segment = []
        
        if segment:
            step = max(1, int(1 / subsample_factor))
            subsampled_segment = segment[::step]
            segments.extend(subsampled_segment)
        
        self.word.tokens = segments
    
    def _crop(self):
        cropped_tokens = []
        for token in self.word.tokens:
            if isinstance(token, Coordinate):
                x = round(token.x)
                y = round(token.y)
                if 0 <= x <= IMG_W - 1 and 0 <= y <= IMG_H - 1:
                    cropped_tokens.append(Coordinate(x=x, y=y))
            else:
                cropped_tokens.append(token)
        self.word.tokens = cropped_tokens

    def _deduplicate(self):
        deduplicated_tokens = []
        for token in self.word.tokens:
            if not deduplicated_tokens:
                deduplicated_tokens.append(token)
            elif token != deduplicated_tokens[-1]:
                deduplicated_tokens.append(token)
        self.word.tokens = deduplicated_tokens
    
    def random_augment(self):
        coordinates = self.word.extract_coordinates()
        normalized_coords = self._normalize_coordinates(coordinates, center=False)
        rescaled_coords = self._random_rescale(normalized_coords)
        sheared_coords = self._random_shear(rescaled_coords)
        translated_coords = self._random_translate(sheared_coords)
        self.word.insert_coordinates(translated_coords)
        self._subsample()
        self._crop()
        self._deduplicate()

    def normalize(self):
        coordinates = self.word.extract_coordinates()
        normalized_coords = self._normalize_coordinates(coordinates)
        self.word.insert_coordinates(normalized_coords)
        self.subsample_factor_range = (0.3, 0.3)
        self._subsample()
        self._crop()
        self._deduplicate()
        
    def visualize(self):
        import matplotlib.pyplot as plt

        # Create a new figure
        plt.figure(figsize=(40, 10))
        
        # Start a new line segment
        current_segment_x = []
        current_segment_y = []
        
        # Plot all coordinates, breaking at Pause tokens
        for token in self.word.tokens:
            if isinstance(token, Coordinate):
                current_segment_x.append(token.x)
                current_segment_y.append(token.y)
            elif isinstance(token, Pause):
                # Plot the current segment if it has points
                if current_segment_x:
                    plt.plot(current_segment_x, current_segment_y, 'b-')
                    # Also plot points to make them visible
                    plt.plot(current_segment_x, current_segment_y, 'r.')
                # Reset segments for the next stroke
                current_segment_x = []
                current_segment_y = []
        
        # Plot the last segment if it exists
        if current_segment_x:
            plt.plot(current_segment_x, current_segment_y, 'b-')
            plt.plot(current_segment_x, current_segment_y, 'r.')
        
        # Set the axis limits and invert y-axis (since image coordinates have origin at top-left)
        plt.xlim(0, IMG_W)
        plt.ylim(IMG_H, 0)
        
        # Add title with the word being visualized
        plt.title(f'Word: {self.word.word}')
        
        # Display the plot
        plt.show()
        

if __name__ == '__main__':
    import json
    raw_coord_path = list(RAW_COORD_FOLDER.iterdir())[24]
    with open(raw_coord_path, 'r') as f:
        data = json.load(f)
    word = Word(data['word'], data['tokens'])
    augmenter = WordAugmenter(word)
    augmenter.random_augment()
    augmenter.visualize()

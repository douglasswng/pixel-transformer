from typing import List
import random
import math
import numpy as np
from dataloader.word_dataclass import Word, Coordinate, Pause
from constants import IMG_H, IMG_W, RAW_COORD_FOLDER

random.seed(42)

def rescale(coordinates: List[Coordinate], scale_factor_x: float, scale_factor_y: float) -> List[Coordinate]:
    center_x = IMG_W / 2
    center_y = IMG_H / 2
    rescaled_coords = []
    
    for coord in coordinates:
        offset_x = coord.x - center_x
        offset_y = coord.y - center_y
        
        scaled_offset_x = offset_x * scale_factor_x
        scaled_offset_y = offset_y * scale_factor_y
        
        new_x = center_x + scaled_offset_x
        new_y = center_y + scaled_offset_y
        
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
        self.shear_factor_range = (-0.3, 0.3)
        self.normalize_padding = 5
        self.rescale_factor_range = (0.7, 1)
        self.translate_range = (-5, 5)
        self.subsample_step_range = (1, 2)

    def _random_shear(self, coordinates: List[Coordinate]) -> List[Coordinate]:
        shear_factor = random.uniform(*self.shear_factor_range)
        return shear(coordinates, shear_factor)

    def _normalize_coordinates(self, coordinates: List[Coordinate], center: bool = True) -> List[Coordinate]:
        min_x = min(coord.x for coord in coordinates)
        min_y = min(coord.y for coord in coordinates)
        max_x = max(coord.x for coord in coordinates)
        max_y = max(coord.y for coord in coordinates)

        new_min_x, new_min_y = self.normalize_padding, self.normalize_padding
        new_max_x, new_max_y = IMG_W - self.normalize_padding - 1, IMG_H - self.normalize_padding - 1

        scale_x = (new_max_x - new_min_x) / (max_x - min_x) if max_x != min_x else math.inf
        scale_y = (new_max_y - new_min_y) / (max_y - min_y) if max_y != min_y else math.inf
        scale = min(scale_x, scale_y)

        if scale == math.inf:
            new_x = (new_min_x + new_max_x) / 2 if center else random.uniform(new_min_x, new_max_x)
            new_y = (new_min_y + new_max_y) / 2 if center else random.uniform(new_min_y, new_max_y)
            return [Coordinate(x=new_x, y=new_y)] * len(coordinates)

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

        return [Coordinate(x=scale * (coord.x - min_x) + offset_x,
                        y=scale * (coord.y - min_y) + offset_y)
                for coord in coordinates]
    
    def _random_rescale(self, coordinates: List[Coordinate]) -> List[Coordinate]:
        rescale_factor_x = random.uniform(*self.rescale_factor_range)
        rescale_factor_y = random.uniform(*self.rescale_factor_range)
        return rescale(coordinates, rescale_factor_x, rescale_factor_y)
    
    def _random_translate(self, coordinates: List[Coordinate]) -> List[Coordinate]:
        shift_x = random.uniform(*self.translate_range)
        shift_y = random.uniform(*self.translate_range)
        return translate(coordinates, shift_x, shift_y)
    
    def _subsample(self):
        step = random.uniform(*self.subsample_step_range)
        segments = []
        segment = []
        
        for token in self.word.tokens:
            if isinstance(token, Coordinate):
                segment.append(token)
            else:
                if segment:
                    segments.extend(self._subsample_segment(segment, step))
                segments.append(token)
                segment = []
        
        if segment:
            segments.extend(self._subsample_segment(segment, step))
        
        self.word.tokens = segments

    def _subsample_segment(self, segment: List[Coordinate], step: float):
        if len(segment) <= 2:
            return segment
        
        indices = np.round(np.arange(0, len(segment) - 1, step)).astype(int)
        indices = np.unique(np.append(indices, len(segment) - 1))
        
        return [segment[i] for i in indices]

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
        sheared_coords = self._random_shear(coordinates)
        normalized_coords = self._normalize_coordinates(sheared_coords, center=False)
        rescaled_coords = self._random_rescale(normalized_coords)
        translated_coords = self._random_translate(rescaled_coords)
        self.word.insert_coordinates(translated_coords)
        self._subsample()
        self._deduplicate()

    def normalize(self):
        coordinates = self.word.extract_coordinates()
        normalized_coords = self._normalize_coordinates(coordinates)
        self.word.insert_coordinates(normalized_coords)
        self.subsample_step_range = ((self.subsample_step_range[0] + self.subsample_step_range[1]) / 2,) * 2
        self._subsample()
        self._deduplicate()
        
    def visualize(self):
        import matplotlib.pyplot as plt

        plt.figure(figsize=(40, 10))
        
        current_segment_x = []
        current_segment_y = []
        
        for token in self.word.tokens:
            if isinstance(token, Coordinate):
                current_segment_x.append(token.x)
                current_segment_y.append(token.y)
            elif isinstance(token, Pause):
                if current_segment_x:
                    plt.plot(current_segment_x, current_segment_y, 'b-')
                    plt.plot(current_segment_x, current_segment_y, 'r.')
                current_segment_x = []
                current_segment_y = []
        
        if current_segment_x:
            plt.plot(current_segment_x, current_segment_y, 'b-')
            plt.plot(current_segment_x, current_segment_y, 'r.')
        
        plt.xlim(0, IMG_W)
        plt.ylim(IMG_H, 0)
        
        plt.title(f'Word: {self.word.word}')
        
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
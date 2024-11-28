from dataclasses import dataclass
from typing import List, Union
from constants import START_TOKEN, PAUSE_TOKEN, END_TOKEN

@dataclass
class Coordinate:
    x: Union[int, float]
    y: Union[int, float]

@dataclass
class Start:
    type: str = START_TOKEN

@dataclass
class Pause:
    type: str = PAUSE_TOKEN

@dataclass
class End:
    type: str = END_TOKEN

class Word:
    def __init__(self, word: str, tokens: List[dict]):
        self.word = word
        self.tokens = [self._parse_token(token) for token in tokens]

    def _parse_token(self, token: dict) -> Union[Coordinate, Start, Pause, End]:
        if token['type'] == 'coordinate':
            x, y = token['x'], token['y']
            return Coordinate(x=x, y=y)
        elif token['type'] == 'start':
            return Start()
        elif token['type'] == 'pause':
            return Pause()
        elif token['type'] == 'end':
            return End()
        else:
            raise ValueError(f'Unknown token type: {token["type"]}')
        
    def extract_coordinates(self) -> List[Coordinate]:
        return [token for token in self.tokens if isinstance(token, Coordinate)]

    def insert_coordinates(self, coordinates: List[Coordinate]) -> None:
        coord_idx = 0
        for i in range(len(self.tokens)):
            if isinstance(self.tokens[i], Coordinate):
                if coord_idx < len(coordinates):
                    self.tokens[i] = coordinates[coord_idx]
                    coord_idx += 1
                else:
                    raise ValueError("Not enough coordinates provided to fill all coordinate positions")
        
        if coord_idx < len(coordinates):
            raise ValueError("Too many coordinates provided")

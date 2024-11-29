from typing import List, Union, Optional
import io
import PIL.Image as Image
import matplotlib.pyplot as plt
from dataloader.word_dataclass import Word
from dataloader.word_dataclass import Coordinate, Start, Pause, End
from constants import IMG_W, IMG_H, SPECIAL_TOKEN_COUNT, TOKEN_COUNT, PAD_ID, START_ID, MOVE_ID, DRAW_ID, END_ID

def to_id(word: Word, target_len: Optional[int] = None) -> List[int]:
    token_ids = []
    for idx, token in enumerate(word.tokens):
        if isinstance(token, Start):
            token_ids.append(START_ID)
        elif isinstance(token, End):
            token_ids.append(END_ID)
        elif isinstance(token, Coordinate):
            if idx > 0 and isinstance(word.tokens[idx-1], Pause):
                token_ids.append(MOVE_ID)
            else:
                token_ids.append(DRAW_ID)
            x, y = int(token.x), int(token.y)
            token_id = x + IMG_W * y + SPECIAL_TOKEN_COUNT
            token_ids.append(token_id)
        elif isinstance(token, Pause):
            pass
        else:
            raise ValueError(f"Unexpected token type: {type(token)}")

    for id in token_ids:
        if id > TOKEN_COUNT - 1:
            raise ValueError(f"Token ID {id} exceeds the maximum allowed value of {TOKEN_COUNT}")
        
    if target_len is None:
        return token_ids
    else:     
        return token_ids + [PAD_ID] * (target_len - len(token_ids))

def to_tokens(ids: List[int]) -> List[Union[Coordinate, Start, Pause, End]]:
    tokens = []
    draw_mode = True
    for id in ids:
        if id == PAD_ID:
            continue
        elif id == START_ID:
            tokens.append(Start())
        elif id == END_ID:
            tokens.append(End())
        elif id == MOVE_ID:
            draw_mode = False
            tokens.append(Pause())
        elif id == DRAW_ID:
            draw_mode = True
        else:
            adjusted_id = id - SPECIAL_TOKEN_COUNT
            x = adjusted_id % IMG_W
            y = adjusted_id // IMG_W
            tokens.append(Coordinate(x=x, y=y))
            if not draw_mode:
                draw_mode = True
    return tokens

def draw_tokens(tokens: List[Union[Coordinate, Start, Pause, End]]) -> Image.Image:
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.set_xlim(0, IMG_W)
    ax.set_ylim(IMG_H, 0)
    ax.axis('off')

    fig.patch.set_facecolor('white')

    current_path = []
    for token in tokens:
        if isinstance(token, Coordinate):
            current_path.append((token.x, token.y))
        elif isinstance(token, Pause):
            if current_path:
                x, y = zip(*current_path)
                ax.plot(x, y, 'k-', linewidth=2)
                current_path = []

    if current_path:
        x, y = zip(*current_path)
        ax.plot(x, y, 'k-', linewidth=2)

    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight', pad_inches=0, facecolor='white')
    buf.seek(0)
    img = Image.open(buf)
    
    border_size = 10
    bordered_img = Image.new('RGB', (img.width + 2*border_size, img.height + 2*border_size), color='lightgrey')
    bordered_img.paste(img, (border_size, border_size))

    plt.close(fig)
    return bordered_img

if __name__ == '__main__':
    import json
    from dataloader.word_dataclass import Word
    from constants import RAW_COORD_FOLDER
    word_path = next(RAW_COORD_FOLDER.iterdir())
    with open(word_path, 'r') as f:
        word_data = json.load(f)
    word, tokens = word_data['word'], word_data['tokens']
    word = Word(word=word, tokens=tokens)
    print(to_id(word, 100)) # doesnt work before normalising
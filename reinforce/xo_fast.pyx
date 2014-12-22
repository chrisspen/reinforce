
def flatten_board(b):
    """
    Converts a 2D list to 1D.
    """
    assert len(b) == 3 and len(b[0]) == 3
    return b[0] + b[1] + b[2]
    
def expand_board(b):
    """
    Converts a 1D list to 2D.
    """
    assert len(b) == 9
    return [b[0:3], b[3:6], b[6:9]]
    
def transform_board(b, as_tuples=False):
    """
    Returns all equivalent board representations.
    
    Typically about 24% faster than the normal Python version
    even though the code is identical.
    """
    
    new_boards = [b]
    
    b = expand_board(b)
    
    # Flip-x.
    new_boards.append(flatten_board([list(reversed(_)) for _ in b]))
    
    # Flip-y.
    new_boards.append(flatten_board([list(_) for _ in list(reversed(b))]))
    
    #http://stackoverflow.com/q/42519/247542
    # Rotate-90.
    rot90 = [list(_) for _ in zip(*b[::-1])]
    new_boards.append(flatten_board(rot90))
    
    # Rotate-180.
    rot180 = [list(_) for _ in zip(*rot90[::-1])]
    new_boards.append(flatten_board(rot180))
    
    # Rotate-270.
    rot270 = [list(_) for _ in zip(*rot180[::-1])]
    new_boards.append(flatten_board(rot270))
    
    # Transpose from top-left to bottom-right.
    new_boards.append(flatten_board([list(_) for _ in zip(*b)]))
    
    # Transpose from top-right to bottom-left.
    new_boards.append(flatten_board([list(_) for _ in list(reversed(rot90))]))
    
    if as_tuples:
        new_boards = [tuple(_) for _ in new_boards]
    
    return new_boards

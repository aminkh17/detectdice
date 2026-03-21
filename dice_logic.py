import math

def calculate_index(r: int, g: int, b: int) -> int:
    """Index = (R−1)×36 + (G−1)×6 + B"""
    return (r - 1) * 36 + (g - 1) * 6 + b

def calculate_final(index: int, topnumber: int = 35) -> int:
    """Final = 1 + floor( (Index−1) × (topnumber−1) / 215 )"""
    # Using (topnumber - 1) in the formula as implied by the default 35 -> 34
    return 1 + math.floor((index - 1) * (topnumber - 1) / 215)
from carddeck import *

def player_has_ace(cards: []):
    has_ace = [card for card in cards if Rank.ACE == card.rank]
    return (has_ace.__len__() > 0)
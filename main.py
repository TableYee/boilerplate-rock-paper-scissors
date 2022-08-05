# This entrypoint file to be used in development. Start by reading README.md
import import_ipynb
from RPS_game import play, mrugesh, abbey, quincy, kris, human, random_player
from RPS import player
from unittest import main

# play(player, mrugesh, 3000)
# play(player, quincy, 3000)
# play(player, abbey, 3000)
# play(player, kris, 3000)

# Uncomment line below to play interactively against a bot:
play(human, player, 20, verbose=True)

# Uncomment line below to play against a bot that plays randomly:
# play(human, random_player, 1000)



# Uncomment line below to run unit tests automatically
# main(module='test_module', exit=False)
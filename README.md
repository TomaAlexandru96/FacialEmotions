# FacialEmotions
usage: main.py [-h] (--load | --train) [--save] [--data DATA] [--dump]

optional arguments:
  -h, --help   show this help message and exit
  --load       Load the trees.
  --train      Train the trees.
  --save       Train and save the trees.
  --data DATA  What kind of date do you want to use clean or noisy
  --dump       Print the trees.

  example: python3 main.py --load
  Loads the trees, test the trees with loaded data and prints accuracy 

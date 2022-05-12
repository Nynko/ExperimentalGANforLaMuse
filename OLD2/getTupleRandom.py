import os
import random


def getTupleRandom():
   listCollages = os.listdir("./Images/Collage")
   listReal = os.listdir("./Images/Real")
   random.shuffle(listCollages)
   random.shuffle(listReal)
   
   n = min(len(listCollages), len(listReal))
   
   tuples = []
   
   for i in range(n):
      tuples.append( (listCollages[i], listReal[i]) )
   
   return tuples


if __name__ == "__main__":
   print(getTupleRandom())
   
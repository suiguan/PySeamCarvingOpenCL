from PIL import Image
import numpy as np
import sys
import time

class SeamCarver:
   def __init__(self, img): #assume the image is RGB mode
      self.reset(img)

   def reset(self, img):
      self.img = img
      self.width, self.height = self.img.size
      r, g, b = self.img.split()
      self.r = r.load()
      self.g = g.load()
      self.b = b.load()

   #return the dual gradient energy at given pixel
   def getDualGradientEnergy(self, x, y):
      #first calcuate x-energy from r, g, b different between (x-1,y) and (x+1,y)
      x1 = x - 1 if x > 0 else self.width - 1
      x2 = x + 1 if x < self.width - 1 else 0
      rx = abs(self.r[x1, y] - self.r[x2, y])
      gx = abs(self.g[x1, y] - self.g[x2, y])
      bx = abs(self.b[x1, y] - self.b[x2, y])
      x_energy = (rx**2) + (gx**2) + (bx**2)

      #then calcuate y-energy from r, g, b different between (x,y-1) and (x,y+1)
      y1 = y - 1 if y > 0 else self.height - 1
      y2 = y + 1 if y < self.height - 1 else 0
      ry = abs(self.r[x, y1] - self.r[x, y2])
      gy = abs(self.g[x, y1] - self.g[x, y2])
      by = abs(self.b[x, y1] - self.b[x, y2])
      y_energy = (ry**2) + (gy**2) + (by**2)
      
      #return the total energy
      return x_energy + y_energy

   def getEnergyMap(self):
      self.energy = [] 
      for y in range(0, self.height):
         for x in range(0, self.width):
            self.energy.append(self.getDualGradientEnergy(x,y))

   def getEnergy(self, x, y):
      return self.energy[x+(y*self.width)]

   def setEnergy(self, x, y, v):
      self.energy[x+(y*self.width)] = v

   #return a length height array of the pixel x-index of the vertical seam
   def findVerticalSeam(self):
      before = int(time.time())
      self.getEnergyMap()
      after = int(time.time())
      #print("Take %d seconds to build energy map" % (after-before))
      findMin = False
      minEn = sys.maxint
      bottom_x = 0
      for y in range(0, self.height):
         for x in range(0, self.width):
            if y > 0:
               if not findMin and y == self.height - 1: findMin = True
               if x - 1 < 0:
                  en = self.getEnergy(x,y) + min(self.getEnergy(x,y-1), self.getEnergy(x+1,y-1))
               elif x + 1 >= self.width:
                  en = self.getEnergy(x,y) + min(self.getEnergy(x-1,y-1), self.getEnergy(x,y-1))
               else:
                  en = self.getEnergy(x,y) + min(self.getEnergy(x-1,y-1), self.getEnergy(x,y-1), self.getEnergy(x+1,y-1))
               self.setEnergy(x, y, en) #dynamic program
               if findMin and en < minEn:
                  minEn = en
                  bottom_x = x 
      return self.findVerticalSeamFromBottomX(bottom_x)

   def findVerticalSeamFromBottomX(self, x):
      seam = [x,]
      cur_x = x
      for y in reversed(range(0, self.height-1)):
         if cur_x - 1 < 0:
            min_en = min(self.getEnergy(cur_x, y), self.getEnergy(cur_x+1, y))
            if min_en == self.getEnergy(cur_x+1,y): 
               seam.append(cur_x+1)
               cur_x = cur_x+1
            else:
	       seam.append(cur_x)
         elif cur_x + 1 >= self.width:
            min_en = min(self.getEnergy(cur_x-1, y), self.getEnergy(cur_x, y))
            if min_en == self.getEnergy(cur_x-1,y): 
               seam.append(cur_x-1)
               cur_x = cur_x-1
            else:
	       seam.append(cur_x)
         else:
            min_en = min(self.getEnergy(cur_x-1, y), self.getEnergy(cur_x,y), self.getEnergy(cur_x+1, y))
            if min_en == self.getEnergy(cur_x-1,y): 
               seam.append(cur_x-1)
               cur_x = cur_x-1
            elif min_en == self.getEnergy(cur_x+1,y): 
               seam.append(cur_x+1)
               cur_x = cur_x+1
            else: 
	       seam.append(cur_x)
      seam.reverse()
      return seam

   #remove the given vertical seam index
   def removeVerticalSeam(self, vseam):
      if len(vseam) != self.height:
         print("invalid seam length %d, height %d" % (len(vseam), self.height))
         return
      img_ar = np.array(self.img)
      new_img_ar = np.array([np.delete(img_ar[y], vseam[y], axis=0) for y in range(0, self.height)])
      self.reset(Image.fromarray(new_img_ar))

   def dumpImg(self, name):
      self.img.save(name, "JPEG")




#---------------- Main ---------------
def usage(prog):
   print("Usage: python %s <jpeg>" % prog)
   sys.exit(-1);

def main(argv):
   if len(argv) != 2: usage(argv[0])
   img = Image.open(argv[1])
   w,h = img.size
   new_w = w/2

   #regular resizing
   img.resize((new_w, h), Image.BILINEAR).save("bilinear_resize_%dx%d.jpg" % (new_w, h), "JPEG") 

   #seam carving resizing
   carver = SeamCarver(img)
   for i in range(0, new_w):
      print("iteration = %d, new_w = %d" % (i, new_w))
      before = int(time.time())
      vseam = carver.findVerticalSeam()
      afterFindSeam = int(time.time())
      #print("Take %d seconds to find seam" % (afterFindSeam - before))
      carver.removeVerticalSeam(vseam)
      afterRemoveSeam = int(time.time())
      #print("Take %d seconds to remove seam" % (afterRemoveSeam - afterFindSeam))

   carver.dumpImg("out.jpg")

if __name__ == "__main__":
   main(sys.argv)

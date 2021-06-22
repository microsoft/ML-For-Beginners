# Maze simulation environment for Reinforcement Learning tutorial
# by Dmitry Soshnikov
# http://soshnikov.com

import matplotlib.pyplot as plt
import numpy as np
import cv2
import random
import math

def clip(min,max,x):
    if x<min:
        return min
    if x>max:
        return max
    return x

def imload(fname,size):
    img = cv2.imread(fname)
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    img = cv2.resize(img,(size,size),interpolation=cv2.INTER_LANCZOS4)
    img = img / np.max(img)
    return img

def draw_line(dx,dy,size=50):
    p=np.ones((size-2,size-2,3))
    if dx==0:
        dx=0.001
    m = (size-2)//2
    l = math.sqrt(dx*dx+dy*dy)*(size-4)/2
    a = math.atan(dy/dx)
    cv2.line(p,(int(m-l*math.cos(a)),int(m-l*math.sin(a))),(int(m+l*math.cos(a)),int(m+l*math.sin(a))),(0,0,0),1)
    s = -1 if dx<0 else 1
    cv2.circle(p,(int(m+s*l*math.cos(a)),int(m+s*l*math.sin(a))),3,0)
    return p   

def probs(v):
    v = v-v.min()
    if (v.sum()>0):
        v = v/v.sum()
    return v

class Board:
    class Cell:
        empty = 0
        water = 1
        wolf = 2
        tree = 3
        apple = 4
    def __init__(self,width,height,size=50):
        self.width = width
        self.height = height
        self.size = size+2
        self.matrix = np.zeros((width,height))
        self.grid_color = (0.6,0.6,0.6)
        self.background_color = (1.0,1.0,1.0)
        self.grid_thickness = 1
        self.grid_line_type = cv2.LINE_AA
        self.pics = {
            "wolf" : imload('images/wolf.png',size-4),
            "apple" : imload('images/apple.png',size-4),
            "human" : imload('images/human.png',size-4)
        }
        self.human = (0,0)
        self.frame_no = 0

    def randomize(self,water_size=5, num_water=3, num_wolves=1, num_trees=5, num_apples=3,seed=None):
        if seed:
            random.seed(seed)
        for _ in range(num_water):
            x = random.randint(0,self.width-1)
            y = random.randint(0,self.height-1)
            for _ in range(water_size):
                self.matrix[x,y] = Board.Cell.water
                x = clip(0,self.width-1,x+random.randint(-1,1))
                y = clip(0,self.height-1,y+random.randint(-1,1))
        for _ in range(num_trees):
            while True:
                x = random.randint(0,self.width-1)
                y = random.randint(0,self.height-1)
                if self.matrix[x,y]==Board.Cell.empty:
                    self.matrix[x,y] = Board.Cell.tree # tree
                    break
        for _ in range(num_wolves):
            while True:
                x = random.randint(0,self.width-1)
                y = random.randint(0,self.height-1)
                if self.matrix[x,y]==Board.Cell.empty:
                    self.matrix[x,y] = Board.Cell.wolf # wolf
                    break
        for _ in range(num_apples):
            while True:
                x = random.randint(0,self.width-1)
                y = random.randint(0,self.height-1)
                if self.matrix[x,y]==Board.Cell.empty:
                    self.matrix[x,y] = Board.Cell.apple
                    break

    def at(self,pos=None):
        if pos:
            return self.matrix[pos[0],pos[1]]
        else:
            return self.matrix[self.human[0],self.human[1]]

    def is_valid(self,pos):
        return pos[0]>=0 and pos[0]<self.width and pos[1]>=0 and pos[1] < self.height

    def move_pos(self, pos, dpos):
        return (pos[0] + dpos[0], pos[1] + dpos[1])

    def move(self,dpos,check_correctness=True):
        new_pos = self.move_pos(self.human,dpos)
        if self.is_valid(new_pos) or not check_correctness:
            self.human = new_pos

    def random_pos(self):
        x = random.randint(0,self.width-1)
        y = random.randint(0,self.height-1)
        return (x,y)

    def random_start(self):
        while True:
            pos = self.random_pos()
            if self.at(pos) == Board.Cell.empty:
                self.human = pos
                break


    def image(self,Q=None):
        img = np.zeros((self.height*self.size+1,self.width*self.size+1,3))
        img[:,:,:] = self.background_color
        # Draw water
        for x in range(self.width):
            for y in range(self.height):
                if (x,y) == self.human:
                    ov = self.pics['human']
                    img[self.size*y+2:self.size*y+ov.shape[0]+2,self.size*x+2:self.size*x+2+ov.shape[1],:] = np.minimum(ov,1.0)
                    continue
                if self.matrix[x,y] == Board.Cell.water:
                    img[self.size*y:self.size*(y+1),self.size*x:self.size*(x+1),:] = (0,0,1.0)
                if self.matrix[x,y] == Board.Cell.wolf:
                    ov = self.pics['wolf']
                    img[self.size*y+2:self.size*y+ov.shape[0]+2,self.size*x+2:self.size*x+2+ov.shape[1],:] = np.minimum(ov,1.0)
                if self.matrix[x,y] == Board.Cell.apple: # apple
                    ov = self.pics['apple']
                    img[self.size*y+2:self.size*y+ov.shape[0]+2,self.size*x+2:self.size*x+2+ov.shape[1],:] = np.minimum(ov,1.0)
                if self.matrix[x,y] == Board.Cell.tree: # tree
                    img[self.size*y:self.size*(y+1),self.size*x:self.size*(x+1),:] = (0,1.0,0)
                if self.matrix[x,y] == Board.Cell.empty and Q is not None:
                    p = probs(Q[x,y])
                    dx,dy = 0,0
                    for i,(ddx,ddy) in enumerate([(-1,0),(1,0),(0,-1),(0,1)]):
                        dx += ddx*p[i]
                        dy += ddy*p[i]
                        l = draw_line(dx,dy,self.size)
                        img[self.size*y+2:self.size*y+l.shape[0]+2,self.size*x+2:self.size*x+2+l.shape[1],:] = l

        # Draw grid
        for i in range(self.height+1):
            img[:,i*self.size] = 0.3
            #cv2.line(img,(0,i*self.size),(self.width*self.size,i*self.size), self.grid_color, self.grid_thickness,lineType=self.grid_line_type)
        for j in range(self.width+1):
            img[j*self.size,:] = 0.3
            #cv2.line(img,(j*self.size,0),(j*self.size,self.height*self.size), self.grid_color, self.grid_thickness,lineType=self.grid_line_type)
        return img

    def plot(self,Q=None):
        plt.figure(figsize=(11,6))
        plt.imshow(self.image(Q),interpolation='hanning')

    def saveimage(self,filename,Q=None):
        cv2.imwrite(filename,255*self.image(Q)[...,::-1])

    def walk(self,policy,save_to=None,start=None):
        n = 0
        if start:
            self.human = start
        else:
            self.random_start()

        while True:
            if save_to:
                self.saveimage(save_to.format(self.frame_no))
                self.frame_no+=1
            if self.at() == Board.Cell.apple:
                return n # success!
            if self.at() in [Board.Cell.wolf, Board.Cell.water]:
                return -1 # eaten by wolf or drowned
            while True:
                a = policy(self)
                new_pos = self.move_pos(self.human,a)
                if self.is_valid(new_pos) and self.at(new_pos)!=Board.Cell.water:
                    self.move(a) # do the actual move
                    break
            n+=1
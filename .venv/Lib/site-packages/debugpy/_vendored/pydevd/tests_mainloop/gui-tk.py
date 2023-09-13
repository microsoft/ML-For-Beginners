#!/usr/bin/env python
"""Simple Tk example to manually test event loop integration.

To run this:
1) Enable the PyDev GUI event loop integration for tk
2) do an execfile on this script
3) ensure you have a working GUI simultaneously with an
   interactive console
"""

if __name__ == '__main__':
    
    try:
        from Tkinter import *
    except:
        # Python 3
        from tkinter import *
    
    class MyApp:
    
        def __init__(self, root):
            frame = Frame(root)
            frame.pack()
    
            self.button = Button(frame, text="Hello", command=self.hello_world)
            self.button.pack(side=LEFT)
    
        def hello_world(self):
            print("Hello World!")
    
    root = Tk()
    
    app = MyApp(root)

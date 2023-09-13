#!/usr/bin/env python
"""Simple pyglet example to manually test event loop integration.

To run this:
1) Enable the PyDev GUI event loop integration for pyglet
2) do an execfile on this script
3) ensure you have a working GUI simultaneously with an
   interactive console
"""

if __name__ == '__main__':
    import pyglet
    
    
    window = pyglet.window.Window()
    label = pyglet.text.Label('Hello, world',
                              font_name='Times New Roman',
                              font_size=36,
                              x=window.width//2, y=window.height//2,
                              anchor_x='center', anchor_y='center')
    @window.event
    def on_close():
        window.close()
    
    @window.event
    def on_draw():
        window.clear()
        label.draw()

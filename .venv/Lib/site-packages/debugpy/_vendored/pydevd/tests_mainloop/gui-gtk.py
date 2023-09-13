#!/usr/bin/env python
"""Simple GTK example to manually test event loop integration.

To run this:
1) Enable the PyDev GUI event loop integration for gtk
2) do an execfile on this script
3) ensure you have a working GUI simultaneously with an
   interactive console
"""

if __name__ == '__main__':
    import pygtk
    pygtk.require('2.0')
    import gtk
    
    
    def hello_world(wigdet, data=None):
        print("Hello World")
    
    def delete_event(widget, event, data=None):
        return False
    
    def destroy(widget, data=None):
        gtk.main_quit()
    
    window = gtk.Window(gtk.WINDOW_TOPLEVEL)
    window.connect("delete_event", delete_event)
    window.connect("destroy", destroy)
    button = gtk.Button("Hello World")
    button.connect("clicked", hello_world, None)
    
    window.add(button)
    button.show()
    window.show()
    

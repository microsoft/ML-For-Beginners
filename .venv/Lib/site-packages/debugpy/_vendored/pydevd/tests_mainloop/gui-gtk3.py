#!/usr/bin/env python
"""Simple Gtk example to manually test event loop integration.

To run this:
1) Enable the PyDev GUI event loop integration for gtk3
2) do an execfile on this script
3) ensure you have a working GUI simultaneously with an
   interactive console
"""

if __name__ == '__main__':
    from gi.repository import Gtk
    
    
    def hello_world(wigdet, data=None):
        print("Hello World")
    
    def delete_event(widget, event, data=None):
        return False
    
    def destroy(widget, data=None):
        Gtk.main_quit()
    
    window = Gtk.Window(Gtk.WindowType.TOPLEVEL)
    window.connect("delete_event", delete_event)
    window.connect("destroy", destroy)
    button = Gtk.Button("Hello World")
    button.connect("clicked", hello_world, None)
    
    window.add(button)
    button.show()
    window.show()
    

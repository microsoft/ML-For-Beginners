#!/usr/bin/env python
"""
A Simple wx example to test PyDev's event loop integration.

To run this:
1) Enable the PyDev GUI event loop integration for wx
2) do an execfile on this script
3) ensure you have a working GUI simultaneously with an
   interactive console

Ref: Modified from wxPython source code wxPython/samples/simple/simple.py
"""

if __name__ == '__main__':

    import wx
    
    
    class MyFrame(wx.Frame):
        """
        This is MyFrame.  It just shows a few controls on a wxPanel,
        and has a simple menu.
        """
        def __init__(self, parent, title):
            wx.Frame.__init__(self, parent, -1, title,
                              pos=(150, 150), size=(350, 200))
    
            # Create the menubar
            menuBar = wx.MenuBar()
    
            # and a menu
            menu = wx.Menu()
    
            # add an item to the menu, using \tKeyName automatically
            # creates an accelerator, the third param is some help text
            # that will show up in the statusbar
            menu.Append(wx.ID_EXIT, "E&xit\tAlt-X", "Exit this simple sample")
    
            # bind the menu event to an event handler
            self.Bind(wx.EVT_MENU, self.on_time_to_close, id=wx.ID_EXIT)
    
            # and put the menu on the menubar
            menuBar.Append(menu, "&File")
            self.SetMenuBar(menuBar)
    
            self.CreateStatusBar()
    
            # Now create the Panel to put the other controls on.
            panel = wx.Panel(self)
    
            # and a few controls
            text = wx.StaticText(panel, -1, "Hello World!")
            text.SetFont(wx.Font(14, wx.SWISS, wx.NORMAL, wx.BOLD))
            text.SetSize(text.GetBestSize())
            btn = wx.Button(panel, -1, "Close")
            funbtn = wx.Button(panel, -1, "Just for fun...")
    
            # bind the button events to handlers
            self.Bind(wx.EVT_BUTTON, self.on_time_to_close, btn)
            self.Bind(wx.EVT_BUTTON, self.on_fun_button, funbtn)
    
            # Use a sizer to layout the controls, stacked vertically and with
            # a 10 pixel border around each
            sizer = wx.BoxSizer(wx.VERTICAL)
            sizer.Add(text, 0, wx.ALL, 10)
            sizer.Add(btn, 0, wx.ALL, 10)
            sizer.Add(funbtn, 0, wx.ALL, 10)
            panel.SetSizer(sizer)
            panel.Layout()
    
    
        def on_time_to_close(self, evt):
            """Event handler for the button click."""
            print("See ya later!")
            self.Close()
    
        def on_fun_button(self, evt):
            """Event handler for the button click."""
            print("Having fun yet?")
    
    
    class MyApp(wx.App):
        def OnInit(self):
            frame = MyFrame(None, "Simple wxPython App")
            self.SetTopWindow(frame)
    
            print("Print statements go to this stdout window by default.")
    
            frame.Show(True)
            return True
    
    
    if __name__ == '__main__':
    
        app = wx.GetApp()
        if app is None:
            app = MyApp(redirect=False, clearSigInt=False)
        else:
            frame = MyFrame(None, "Simple wxPython App")
            app.SetTopWindow(frame)
            print("Print statements go to this stdout window by default.")
            frame.Show(True)
    

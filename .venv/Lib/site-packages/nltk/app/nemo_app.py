# Finding (and Replacing) Nemo, Version 1.1, Aristide Grange 2006/06/06
# https://aspn.activestate.com/ASPN/Cookbook/Python/Recipe/496783

"""
Finding (and Replacing) Nemo

Instant Regular Expressions
Created by Aristide Grange
"""
import itertools
import re
from tkinter import SEL_FIRST, SEL_LAST, Frame, Label, PhotoImage, Scrollbar, Text, Tk

windowTitle = "Finding (and Replacing) Nemo"
initialFind = r"n(.*?)e(.*?)m(.*?)o"
initialRepl = r"M\1A\2K\3I"
initialText = """\
Lorem ipsum dolor sit amet, consectetur adipisicing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua.
Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat.
Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur.
Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum.
"""
images = {
    "FIND": "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",
    "find": "R0lGODlhMAAiAPQAMf////f39+/v7+fn597e3tbW1s7OzsbGxr29vbW1ta2traWlpZycnJSUlIyMjISEhHt7e3Nzc2tra2NjY1paWlJSUkpKSkJCQjk5OSkpKRgYGAAAAAAAAAAAAAAAAAAAACH+AS4ALAAAAAAwACIAAAX/ICCOZGmeaKquY2AGLiuvMCAUBuHWc48Kh0iFInEYCb4kSQCxPBiMxkMigRQEgJiSFVBYHNGG0RiZOHjblWAiiY4fkDhEYoBp06dAWfyAQyKAgAwDaHgnB0RwgYASgQ0IhDuGJDAIFhMRVFSLEX8QCJJ4AQM5AgQHTZqqjBAOCQQEkWkCDRMUFQsICQ4Vm5maEwwHOAsPDTpKMAsUDlO4CssTcb+2DAp8YGCyNFoCEsZwFQ3QDRTTVBRS0g1QbgsCd5QAAwgIBwYFAwStzQ8UEdCKVchky0yVBw7YuXkAKt4IAg74vXHVagqFBRgXSCAyYWAVCH0SNhDTitCJfSL5/4RbAPKPhQYYjVCYYAvCP0BxEDaD8CheAAHNwqh8MMGPSwgLeJWhwHSjqkYI+xg4MMCEgQjtRvZ7UAYCpghMF7CxONOWJkYR+rCpY4JlVpVxKDwYWEactKW9mhYRtqCTgwgWEMArERSK1j5q//6T8KXonFsShpiJkAECgQYVjykooCVA0JGHEWNiYCHThTFeb3UkoiCCBgwGEKQ1kuAJlhFwhA71h5SukwUM5qqeCSGBgicEWkfNiWSERtBad4JNIBaQBaQah1ToyGZBAnsIuIJs1qnqiAIVjIE2gnAB1T5x0icgzXT79ipgMOOEH6HBbREBMJCeGEY08IoLAkzB1YYFwjxwSUGSNULQJnNUwRYlCcyEkALIxECAP9cNMMABYpRhy3ZsSLDaR70oUAiABGCkAxowCGCAAfDYIQACXoElGRsdXWDBdg2Y90IWktDYGYAB9PWHP0PMdFZaF07SQgAFNDAMAQg0QA1UC8xoZQl22JGFPgWkOUCOL1pZQyhjxinnnCWEAAA7",
    "REPL": "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",
    "repl": "R0lGODlhMAAjAPQAMf////f39+/v7+fn597e3tbW1s7OzsbGxr29vbW1ta2traWlpZycnJSUlIyMjISEhHt7e3Nzc2tra2NjY1paWlJSUkpKSkJCQjk5OTExMSkpKSEhIRgYGBAQEAgICAAAACH+AS4ALAAAAAAwACMAAAX/ICCOZGmeaKqubOu+gCDANBkIQ1EMQhAghFptYEAkEgjEwXBo7ISvweGgWCwUysPjwTgEoCafTySYIhYMxgLBjEQgCULvCw0QdAZdoVhUIJUFChISEAxYeQM1N1OMTAp+UwZ5eA4TEhFbDWYFdC4ECVMJjwl5BwsQa0umEhUVlhESDgqlBp0rAn5nVpBMDxeZDRQbHBgWFBSWDgtLBnFjKwRYCI9VqQsPs0YKEcMXFq0UEalFDWx4BAO2IwPjppAKDkrTWKYUGd7fEJJFEZpM00cOzCgh4EE8SaoWxKNixQooBRMyZMBwAYIRBhUgLDGS4MoBJeoANMhAgQsaCRZm/5lqaCUJhA4cNHjDoKEDBlJUHqkBlYBTiQUZNGjYMMxDhY3VWk6R4MEDBoMUak5AqoYBqANIBo4wcGGDUKIeLlzVZmWJggsVIkwAZaQSA3kdZzlKkIiEAAlDvW5oOkEBs488JTw44oeUIwdvVTFTUK7uiAAPgubt8GFDhQepqETAQCFU1UMGzlqAgFhUsAcCS0AO6lUDhw8xNRSbENGDhgWSHjWUe6ACbKITizmopZoBa6KvOwj9uuHDhwxyj3xekgDDhw5EvWKo0IB4iQLCOCC/njc7ZQ8UeGvza+ABZZgcxJNc4FO1gc0cOsCUrHevc8tdIMTIAhc4F198G2Qwwd8CBIQUAwEINABBBJUwR9R5wElgVRLwWODBBx4cGB8GEzDQIAo33CGJA8gh+JoH/clUgQU0YvDhdfmJdwEFC6Sjgg8yEPAABsPkh2F22cl2AQbn6QdTghTQ5eAJAQyQAAQV0MSBB9gRVZ4GE1mw5JZOAmiAVi1UWcAZDrDyZXYTeaOhA/bIVuIBPtKQ4h7ViYekUPdcEAEbzTzCRp5CADmAAwj+ORGPBcgwAAHo9ABGCYtm0ChwFHShlRiXhmHlkAcCiOeUodqQw5W0oXLAiamy4MOkjOyAaqxUymApDCEAADs=",
}
colors = ["#FF7B39", "#80F121"]
emphColors = ["#DAFC33", "#F42548"]
fieldParams = {
    "height": 3,
    "width": 70,
    "font": ("monaco", 14),
    "highlightthickness": 0,
    "borderwidth": 0,
    "background": "white",
}
textParams = {
    "bg": "#F7E0D4",
    "fg": "#2321F1",
    "highlightthickness": 0,
    "width": 1,
    "height": 10,
    "font": ("verdana", 16),
    "wrap": "word",
}


class Zone:
    def __init__(self, image, initialField, initialText):
        frm = Frame(root)
        frm.config(background="white")
        self.image = PhotoImage(format="gif", data=images[image.upper()])
        self.imageDimmed = PhotoImage(format="gif", data=images[image])
        self.img = Label(frm)
        self.img.config(borderwidth=0)
        self.img.pack(side="left")
        self.fld = Text(frm, **fieldParams)
        self.initScrollText(frm, self.fld, initialField)
        frm = Frame(root)
        self.txt = Text(frm, **textParams)
        self.initScrollText(frm, self.txt, initialText)
        for i in range(2):
            self.txt.tag_config(colors[i], background=colors[i])
            self.txt.tag_config("emph" + colors[i], foreground=emphColors[i])

    def initScrollText(self, frm, txt, contents):
        scl = Scrollbar(frm)
        scl.config(command=txt.yview)
        scl.pack(side="right", fill="y")
        txt.pack(side="left", expand=True, fill="x")
        txt.config(yscrollcommand=scl.set)
        txt.insert("1.0", contents)
        frm.pack(fill="x")
        Frame(height=2, bd=1, relief="ridge").pack(fill="x")

    def refresh(self):
        self.colorCycle = itertools.cycle(colors)
        try:
            self.substitute()
            self.img.config(image=self.image)
        except re.error:
            self.img.config(image=self.imageDimmed)


class FindZone(Zone):
    def addTags(self, m):
        color = next(self.colorCycle)
        self.txt.tag_add(color, "1.0+%sc" % m.start(), "1.0+%sc" % m.end())
        try:
            self.txt.tag_add(
                "emph" + color, "1.0+%sc" % m.start("emph"), "1.0+%sc" % m.end("emph")
            )
        except:
            pass

    def substitute(self, *args):
        for color in colors:
            self.txt.tag_remove(color, "1.0", "end")
            self.txt.tag_remove("emph" + color, "1.0", "end")
        self.rex = re.compile("")  # default value in case of malformed regexp
        self.rex = re.compile(self.fld.get("1.0", "end")[:-1], re.MULTILINE)
        try:
            re.compile("(?P<emph>%s)" % self.fld.get(SEL_FIRST, SEL_LAST))
            self.rexSel = re.compile(
                "%s(?P<emph>%s)%s"
                % (
                    self.fld.get("1.0", SEL_FIRST),
                    self.fld.get(SEL_FIRST, SEL_LAST),
                    self.fld.get(SEL_LAST, "end")[:-1],
                ),
                re.MULTILINE,
            )
        except:
            self.rexSel = self.rex
        self.rexSel.sub(self.addTags, self.txt.get("1.0", "end"))


class ReplaceZone(Zone):
    def addTags(self, m):
        s = sz.rex.sub(self.repl, m.group())
        self.txt.delete(
            "1.0+%sc" % (m.start() + self.diff), "1.0+%sc" % (m.end() + self.diff)
        )
        self.txt.insert("1.0+%sc" % (m.start() + self.diff), s, next(self.colorCycle))
        self.diff += len(s) - (m.end() - m.start())

    def substitute(self):
        self.txt.delete("1.0", "end")
        self.txt.insert("1.0", sz.txt.get("1.0", "end")[:-1])
        self.diff = 0
        self.repl = rex0.sub(r"\\g<\1>", self.fld.get("1.0", "end")[:-1])
        sz.rex.sub(self.addTags, sz.txt.get("1.0", "end")[:-1])


def launchRefresh(_):
    sz.fld.after_idle(sz.refresh)
    rz.fld.after_idle(rz.refresh)


def app():
    global root, sz, rz, rex0
    root = Tk()
    root.resizable(height=False, width=True)
    root.title(windowTitle)
    root.minsize(width=250, height=0)
    sz = FindZone("find", initialFind, initialText)
    sz.fld.bind("<Button-1>", launchRefresh)
    sz.fld.bind("<ButtonRelease-1>", launchRefresh)
    sz.fld.bind("<B1-Motion>", launchRefresh)
    sz.rexSel = re.compile("")
    rz = ReplaceZone("repl", initialRepl, "")
    rex0 = re.compile(r"(?<!\\)\\([0-9]+)")
    root.bind_all("<Key>", launchRefresh)
    launchRefresh(None)
    root.mainloop()


if __name__ == "__main__":
    app()

__all__ = ["app"]

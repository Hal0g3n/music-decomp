from tkinter import *
import tkinter.ttk as ttk
from tkinter.filedialog import askopenfilename as openfile, asksaveasfilename as savefile
from pygame import mixer

# Making the model seen from this file
import sys
sys.path.insert(1, '../')

from AI.SheetMusic.math_model import *


class VerticalScrolledFrame(Frame):
    """A pure Tkinter scrollable frame that actually works!
    * Use the 'interior' attribute to place widgets inside the scrollable frame
    * Construct and pack/place/grid normally
    * This frame only allows vertical scrolling
    """
    def __init__(self, parent, *args, **kw):
        Frame.__init__(self, parent, *args, **kw)            

        # create a canvas object and a vertical scrollbar for scrolling it
        vscrollbar = Scrollbar(self, orient=VERTICAL)
        vscrollbar.pack(fill=Y, side=RIGHT, expand=FALSE)
        canvas = Canvas(self, bd=0, highlightthickness=0,
                        yscrollcommand=vscrollbar.set)
        canvas.pack(side=BOTTOM, fill=BOTH, expand=TRUE)
        vscrollbar.config(command=canvas.yview)

        # reset the view
        canvas.xview_moveto(0)
        canvas.yview_moveto(0)

        # create a frame inside the canvas which will be scrolled with it
        self.interior = interior = Frame(canvas)
        interior_id = canvas.create_window(0, 0, window=interior,
                                           anchor=NW)

        # track changes to the canvas and frame width and sync them,
        # also updating the scrollbar
        def _configure_interior(event):
            # update the scrollbars to match the size of the inner frame
            size = (interior.winfo_reqwidth(), interior.winfo_reqheight())
            canvas.config(scrollregion="0 0 %s %s" % size)
            if interior.winfo_reqwidth() != canvas.winfo_width():
                # update the canvas's width to fit the inner frame
                canvas.config(width=interior.winfo_reqwidth())
        interior.bind('<Configure>', _configure_interior)

        def _configure_canvas(event):
            if interior.winfo_reqwidth() != canvas.winfo_width():
                # update the inner frame's width to fill the canvas
                canvas.itemconfigure(interior_id, width=canvas.winfo_width())
        canvas.bind('<Configure>', _configure_canvas)


class ButtonMenu(ttk.Frame):
    def __init__(self, parent):
        super().__init__(parent)
        self.pack(side=TOP)

        self.parent = parent
        self.root = self._nametowidget(parent.winfo_parent())

        # Variable for file URL
        self.url = StringVar(self, "")

        self.isPaused = BooleanVar(self, False)
        self.isPlaying = BooleanVar(self, False)

        self.openButton = ttk.Button(self, text = "Open Music File", command=self.open)
        self.openButton.pack(side=LEFT)

        self.playButton = ttk.Button(self, text="Play Music", command=self.play)
        self.playButton.pack(side=LEFT)

        self.pauseButton = ttk.Button(self, text="Pause Music", command=self.pause)
        self.pauseButton["state"] = "disabled"
        self.pauseButton.pack(side=LEFT)

        self.stopButton = ttk.Button(self, text="Stop Music", command=self.stop)
        self.stopButton["state"] = "disabled"
        self.stopButton.pack(side=LEFT)

        self.transcribeButton = ttk.Button(self, text="Transcribe", command=self.transcribe)
        self.transcribeButton.pack(side=LEFT)

        self.urlLabel = ttk.Label(self, text="")
        self.urlLabel.pack(side=LEFT, padx=6)


    def open(self):
        # Opens system dialog to select music file
        openlocation = openfile(parent=self.root, title="Select music file to open", filetype = [
            ("music", "*.wav")
        ])

        # Set Path Variable
        self.url.set(openlocation)
        
        self.urlLabel["text"] = openlocation.split("/")[-1]


    def play(self):
        # Plays the audio
        if self.isPaused.get():
            pause()

        elif len(self.url.get()) and not self.isPlaying.get():
            mixer.music.load(self.url.get())
            mixer.music.play()
            self.isPlaying.set(True)
            self.playButton["state"] = "disabled"
            self.pauseButton["state"] = "normal"
            self.stopButton["state"] = "normal"

    def pause(self):
        if self.isPaused.get():
            mixer.music.unpause()
            self.pauseButton["text"] = "Pause Music"
            self.stopButton["text"] = "Stop Music"

        else:
            mixer.music.pause()
            self.pauseButton["text"] = "Unpause Music"
            self.stopButton["text"] = "Restart Music"

        self.isPlaying.set(not self.isPlaying.get())
        self.isPaused.set(not self.isPaused.get())

    def stop(self):
        if self.isPlaying.get():
            mixer.music.fadeout(1000)

        elif self.isPaused.get():
            self.pause()
            mixer.music.fadeout(0)

        self.pauseButton["state"] = "disabled"
        self.stopButton["state"] = "disabled"
        self.playButton["state"] = "normal"

        self.isPlaying.set(False)
        self.isPaused.set(False)

    def transcribe(self):
        # Opens system dialog to select music file
        savelocation = savefile(parent=self.root, title="Select music file to open:", filetypes = [("music", ".mid")], defaultextension = [("MIDI",".mid")])

        audio, rate = librosa.load(self.url.get())
        midi = librosaModel(audio, rate, "Acoustic Grand Piano")
        midi.write(savelocation)
        

class MainFrame(VerticalScrolledFrame):
    def __init__(self, parent, init_url = ''):
        super().__init__(parent)
        self.parent = parent
        self.pack(fill=BOTH, expand=1)

        self.buttonMenu = ButtonMenu(self.interior)
        

# Driver Code
if __name__ == "__main__":
    # Window
    root = Tk()

    # Pygame mixer for audio playing
    mixer.init()

    #messagebox.showinfo(title="Music Transcription", message=f"Transcription Success, saved at {savelocation}", **options)
    # Initialise the main frame
    MainFrame(root)

    # Calls to indicate init done
    root.mainloop()

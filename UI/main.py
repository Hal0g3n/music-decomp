from tkinter import *
import tkinter.ttk as ttk
from tkinter.filedialog import askopenfilename as openfile
#from tkinter.filedialog import saveasfilename as save
from pygame import mixer

# A global variable, storing the data gotten from the model
results = {f"Instrument {i}": "Results" for i in range(20)}

def runModel(url):
    # Should ideally give
    pass

class VerticalScrolledFrame(Frame):
    """A pure Tkinter scrollable frame that actually works!
    * Use the 'interior' attribute to place widgets inside the scrollable frame
    * Construct and pack/place/grid normally
    * This frame only allows vertical scrolling

    (I totally did not copy this online)
    """
    def __init__(self, parent, *args, **kw):
        Frame.__init__(self, parent, *args, **kw)            

        # create a canvas object and a vertical scrollbar for scrolling it
        vscrollbar = Scrollbar(self, orient=VERTICAL)
        vscrollbar.pack(fill=Y, side=RIGHT, expand=FALSE)
        canvas = Canvas(self, bd=0, highlightthickness=0,
                        yscrollcommand=vscrollbar.set)
        canvas.pack(side=LEFT, fill=BOTH, expand=TRUE)
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

class CardView(ttk.Frame):
    def __init__(self, parent, instrument):
        super().__init__(parent)
        self.pack(fill=X)

        # Initialise the label for instrument name
        instrumentLabel = Label(self, text = instrument)
        instrumentLabel.pack(side=TOP, anchor = NW)

        # Initialise the button to save audio
        audioButton = ttk.Button(self, text="Save Audio", command=self.saveAudio)
        audioButton.pack(side=LEFT)
        
        # Initialise the button to save sheet music
        sheetButton = ttk.Button(self, text="Save Sheet", command=self.saveSheet)
        sheetButton.pack(side=LEFT)

    def saveAudio(self): pass
        # save() ...

    def saveSheet(self): pass
        # save() ...


class MainFrame(VerticalScrolledFrame):
    def __init__(self, parent, init_url = ''):
        super().__init__(parent)
        self.pack(fill=BOTH, expand=1)

        # Variable for file URL
        self.url = StringVar(init_url)

        # Button for selecting music file
        openButton = ttk.Button(self.interior, text = "Open Music File", command=self.open)
        openButton.pack(side=TOP)

        # Button for playing the selected music
        playButton = ttk.Button(self.interior, text="Play Music File", command=self.play)
        playButton.pack()

        # Button to start the decomposition
        showButton = ttk.Button(self.interior, text="Decompose Music", command=self.decompose)
        showButton.pack(side=TOP)


    def open(self):
        # Opens system dialog to select music file
        openlocation = openfile(parent=self.parent, title="Select music file to open:")

        # Set Path Variable
        self.url.set(openlocation)


    def play(self):
        # Plays the audio
        mixer.music.load(self.url.get())
        mixer.music.play()
  
    def decompose(self):
        global results
        runModel(self.url.get())

        # For every instrument, create 2 buttons for it
        # This is so the user can save the seperated audio files
        for i in results:
            CardView(self.interior, i)

# Driver Code
if __name__ == "__main__":
    # Window
    root = Tk()

    # Pygame mixer for audio playing
    mixer.init()

    # Some window settings
    root.state('zoomed')

    # Initialise the main frame
    MainFrame(root)

    # Calls to indicate init done
    root.mainloop()
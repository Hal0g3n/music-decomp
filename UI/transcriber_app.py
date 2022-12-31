from tkinter import messagebox
from tkinter import *
import tkinter.ttk as ttk
from tkinter.filedialog import askopenfilename as openfile, asksaveasfilename as savefile
from pygame import mixer
# Making the model seen from this file
import sys
sys.path.insert(1, '../')

from AI.SheetMusic.math_model import *


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

        self.openButton = ttk.Button(
            self, text="Open Music File", command=self.open)
        self.openButton.pack(side=LEFT)

        self.playButton = ttk.Button(
            self, text="Play Music", command=self.play)
        self.playButton.pack(side=LEFT)

        self.stopButton = ttk.Button(
            self, text="Stop Music", command=self.stop)
        self.stopButton["state"] = "disabled"
        self.stopButton.pack(side=LEFT)

        self.showButton = ttk.Button(
            self, text="Transcribe", command=self.transcribe)
        self.showButton.pack(side=LEFT)

        self.urlLabel = ttk.Label(self, text="")
        self.urlLabel.pack(side=LEFT, padx=6)

    def open(self):
        # Opens system dialog to select music file
        openlocation = openfile(parent=self.root, title="Select music file to open", filetype=[
            ("music", "*.wav")
        ])
        if openlocation == "":
            return  # Nothing to open (User cancelled)

        # Set Path Variable
        self.url.set(openlocation)

        # Set music variable
        self.music = mixer.Sound(openlocation)

        # Give some indication of successful loading
        self.urlLabel["text"] = openlocation.split("/")[-1]

    def play(self):
        if len(self.url.get()) and not self.isPlaying.get():
            self.music.play()
            self.isPlaying.set(True)
            self.playButton["state"] = "disabled"
            self.stopButton["state"] = "normal"

    def stop(self):
        if self.isPlaying.get():
            self.music.fadeout(1000)

        self.stopButton["state"] = "disabled"
        self.playButton["state"] = "normal"

        self.isPlaying.set(False)

    def transcribe(self):
        # Opens system dialog to select music file
        savelocation = savefile(parent=self.root, title="Select music file to save:", filetypes = [("music", ".mid")], defaultextension = [("MIDI",".mid")])
        if savelocation == "": return  # Nothing to save (User cancelled)

        audio, rate = librosa.load(self.url.get())
        midi = librosaModel(audio, rate, "Acoustic Grand Piano", prob_threshold = 0.4)
        midi.write(savelocation)

        messagebox.showinfo(title="Music Transcription", message=f"Transcription Success, saved at {savelocation}")

class MainFrame(ttk.Frame):
    def __init__(self, parent, init_url = ''):
        super().__init__(parent)
        self.parent = parent
        self.pack(fill=BOTH, expand=1)

        self.buttonMenu = ButtonMenu(self)
        

# Driver Code
if __name__ == "__main__":
    # Window
    root = Tk()

    # Window settings
    root.title("Music Transcriber")

    # Initialise the pygame mixer
    mixer.init()

    # Initialise the main frame
    MainFrame(root)

    # Calls to indicate init done
    root.mainloop()

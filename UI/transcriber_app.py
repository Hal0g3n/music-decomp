from tkinter import messagebox
from tkinter import *
import tkinter.ttk as ttk
from tkinter.filedialog import askopenfilename as openfile, asksaveasfilename as savefile

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

        self.openButton = ttk.Button(self, text = "Open Music File", command=self.open)
        self.openButton.pack(side=LEFT)

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

    def transcribe(self):
        # Opens system dialog to select music file
        savelocation = savefile(parent=self.root, title="Select music file to save:", filetypes = [("music", ".mid")], defaultextension = [("MIDI",".mid")])

        audio, rate = librosa.load(self.url.get())
        midi = librosaModel(audio, rate, "Acoustic Grand Piano")
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

    # Initialise the main frame
    MainFrame(root)

    # Calls to indicate init done
    root.mainloop()

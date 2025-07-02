import os

def select_structure_file():
    """
    Opens a GUI file dialog for the user to select a PDB or mmCIF file.
    Falls back to CLI input if GUI is unavailable.
    Returns the selected file path (str) or '' if cancelled.
    """
    try:
        import tkinter as tk
        from tkinter import filedialog

        root = tk.Tk()
        root.withdraw()
        root.attributes('-topmost', True)  # Bring dialog to front

        file_path = filedialog.askopenfilename(
            title='Select a PDB or mmCIF file',
            filetypes=[('PDB files', '*.pdb'), ('mmCIF files', '*.cif'), ('All files', '*.*')]
        )
        root.destroy()
        if file_path and os.path.isfile(file_path):
            return file_path
        else:
            print("No file selected or file does not exist.")
            return ''
    except Exception as e:
        print("GUI file dialog unavailable or failed. Reason:", e)
        # Fallback to CLI
        file_path = input("Enter full path to your PDB or mmCIF file: ").strip()
        if os.path.isfile(file_path):
            return file_path
        else:
            print("File not found.")
            return ''

import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Učitaj trenirani model
model = load_model('fruit_model.h5')

# Etikete klasa (po redosledu kojim su se učitavale u treningu)
class_labels = ['banana', 'carambola', 'mango', 'peach', 'pitaya']

# Napravi GUI prozor
root = tk.Tk()
root.title("Klasifikacija Voća - CNN")
root.geometry("500x600")
root.configure(bg="#f2f2f2")

# Label za prikaz slike
image_label = tk.Label(root, bg="#f2f2f2")
image_label.pack(pady=20)

# Label za prikaz predikcije
result_label = tk.Label(root, text="", font=("Helvetica", 16), bg="#f2f2f2")
result_label.pack(pady=10)

# Funkcija za predikciju slike
def predict(img_path):
    img = image.load_img(img_path, target_size=(128, 128))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    prediction = model.predict(img_array)
    class_name = class_labels[np.argmax(prediction)]
    return class_name

# Funkcija za otvaranje slike
def open_image():
    file_path = filedialog.askopenfilename()
    if file_path:
        # Prikaži sliku
        img = Image.open(file_path)
        img = img.resize((250, 250))
        tk_img = ImageTk.PhotoImage(img)
        image_label.config(image=tk_img)
        image_label.image = tk_img

        # Prikaži rezultat predikcije
        predicted_label = predict(file_path)
        result_label.config(text=f"Predikcija: {predicted_label.capitalize()}")

# Dugme za učitavanje slike
btn = tk.Button(root, text="Učitaj sliku", command=open_image, font=("Helvetica", 14), bg="#4CAF50", fg="white", padx=10, pady=5)
btn.pack(pady=20)

root.mainloop()

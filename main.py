from svm_hog.svm_hog_model import train_svm_hog

if __name__ == "__main__":
    train_svm_hog(r"data/data_processing/train")

# Da bi se pokrenuo venv koristiti: .venv/Scripts/activate
# Da bi se pokrenuo trening, koristi se komanda:
# python main.py
# Da bi se pokrenula predikcija, koristi se komanda:
# python predict_svm_hog.py
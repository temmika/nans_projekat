from svm_hog.svm_hog_model import predict_image

if __name__ == "__main__":
    img_path = r"data/test/carambola.jpeg"
    predikcija = predict_image(img_path)
    print(f"Predikcija za sliku: {predikcija}")
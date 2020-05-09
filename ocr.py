#import Image




from PIL import Image
import pytesseract
pytesseract.pytesseract.tesseract_cmd = (
    r"C:\Program Files\Tesseract-OCR\tesseract"
)

def ocr(image):

    im = Image.open(image)

    text = pytesseract.image_to_string(im, lang = 'eng')

    return text

#print(text)

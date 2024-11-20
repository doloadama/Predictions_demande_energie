import qrcode
from PIL import Image

# Lien de partage de votre CV
cv_link = "https://drive.google.com/file/d/1fNGWLXIToIMkTwYbmMnba_9i_NOmgbhh/view?usp=drive_link"

# Générer le QR code
qr = qrcode.QRCode(
    version=1,
    error_correction=qrcode.constants.ERROR_CORRECT_H,
    box_size=10,
    border=4,
)
qr.add_data(cv_link)
qr.make(fit=True)

# Personnalisation (facultative)
img = qr.make_image(fill_color="black", back_color="white")

# Enregistrer l'image
img.save("cv_qr_code1.png")

print("QR code généré et enregistré sous 'cv_qr_code.png'.")

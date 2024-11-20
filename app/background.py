from rembg import remove
from PIL import Image
import io

# Charger l'image
input_image_path = "WhatsApp Image 2024-04-29 at 15.34.52.jpeg"
output_image_path = "image_sans_arriere_plan.png"

# Lire l'image en mode binaire
with open(input_image_path, "rb") as input_file:
    input_data = input_file.read()

# Supprimer l'arrière-plan
output_data = remove(input_data)

# Sauvegarder le résultat
with open(output_image_path, "wb") as output_file:
    output_file.write(output_data)

print(f"Image sans arrière-plan sauvegardée sous : {output_image_path}")

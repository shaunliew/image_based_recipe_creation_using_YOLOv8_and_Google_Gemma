import streamlit as st
from langchain_community.llms import Ollama
from ultralytics import YOLO
import cv2
import numpy as np

st.title("Image-Based Recipe Creation using YOLOv8 & Google Gemma")

llm = Ollama(model="gemma")

# Load the pretrained YOLOv8 model
model = YOLO('food.pt')

# Print detection class for the model
class_names = model.names

# Get the image from the user using Streamlit file uploader
uploaded_file = st.file_uploader("Upload an image for your available food ingredients", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Read the uploaded image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    ori_image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    # Convert the color space from BGR to RGB
    image = cv2.cvtColor(ori_image, cv2.COLOR_BGR2RGB)

    # Display the uploaded image
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Detect ingredients in the image using YOLOv8
    results = model(ori_image)

    # Extract the detected ingredients
    detected_ingredients = []
    for result in results:
        for cls in result.boxes.cls:
            ingredient = str(class_names[cls.item()])
            if ingredient not in detected_ingredients:
                detected_ingredients.append(ingredient)

    # Display the detected ingredients
    st.write("Detected Ingredients:")
    st.write(", ".join(detected_ingredients))

    if st.button("Generate Recipe based on Detected Ingredients"):
        st.write("Generating recipe...")

        # Create a prompt for recipe generation based on the detected ingredients
        prompt = f"""
        Generate a recipe using the following main ingredients: {", ".join(detected_ingredients)}.
        Provide a recipe name and list any additional ingredients needed.
        Make sure the recipe name is descriptive and the instructions are clear and easy to follow.
        Format the recipe as follows:

        **Recipe Name: ....**

        **Ingredients:**
        *   ...
        *   ...
        *   ...

        **Instructions:**
        1.  ...
        2.  ...
        3.  ...
        """

        # Create an empty container to hold the generated recipe
        generated_recipe_container = st.empty()

        # Initialize an empty string to store the generated recipe
        generated_recipe = ""

        # Stream the generated recipe and update the container
        for chunk in llm.stream(prompt):
            generated_recipe += chunk
            generated_recipe_container.markdown(generated_recipe)

        st.write("Enjoy your meal!")
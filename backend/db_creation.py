from pymongo import MongoClient
import os

# Connect to MongoDB (adjust the connection string as needed)
client = MongoClient("mongodb://localhost:27017/")

# Database 1: Storing user credentials
db_users = client["UserDB"]
user_collection = db_users["users"]

# Inserting user credentials
user_data = {
    "username": "atharva",
    "email": "atharva@gmail.com",
    "password": "asp@123"
}

user_collection.insert_one(user_data)
print("User data inserted into UserDB.users")

# Database 2: Storing user prompts
db_prompts = client["PromptDB"]
prompt_collection = db_prompts["user_prompts"]

# Inserting prompt related to the user
prompt = "Create a stylish logo for the text 'AP Agro since 2023'"
username = "atharva"

# Dynamically generate the image name (could be any naming convention)
image_name = f"{username}_logo.png"

# Save the image name in the PromptDB along with the prompt
prompt_data = {
    "username": username,
    "prompt": prompt,
    "image_name": image_name  # Store the image name
}

prompt_collection.insert_one(prompt_data)
print("Prompt and image name data inserted into PromptDB.user_prompts")

# Example: Save the generated image locally (optional step)
# Normally, you'd generate the image using an API or other method and save it to disk.
image_path = os.path.join(os.getcwd(), image_name)
# Assuming you have the image data (e.g., from PIL or other libraries)
# Save it locally (this part can be customized as needed)
with open(image_path, "wb") as img_file:
    # For now, let's simulate saving an empty image file
    img_file.write(b"")  # Replace this with actual image bytes if necessary

print(f"Image saved as {image_name} at {image_path}")

# Close the connection
client.close()



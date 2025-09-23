from transformers import AutoTokenizer, AutoModelForCausalLM 
from PIL import Image

model_name = "Qwen/Qwen-VL-Chat"  # hoặc Qwen/Qwen2-VL-7B-Instruct
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True).eval()

# Lấy text feature
text = "A red car is parked on the street"
text_inputs = tokenizer([text], return_tensors="pt")
text_features = model.get_text_features(**text_inputs)  # (batch, hidden_dim)

# # Lấy image feature
# image = Image.open("example.jpg")
# image_features = model.get_image_features(image)  # (batch, hidden_dim)

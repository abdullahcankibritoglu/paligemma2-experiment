PaliGemma 2 English and Turkish Experiment
A small experiment with PaliGemma 2, prepared as part of my Freya Research Fellows article. I asked questions in English and Turkish about the same images to see whether the model returned the correct information and followed the requested language.
The notebook uses google/paligemma2-10b-mix-448 without additional fine-tuning. The Mix checkpoint has already been fine-tuned on several tasks.
Experiment
I used three images and prepared two questions for each image. Each question was asked in both languages, giving 12 responses from six question pairs.
- Car: What color is it, and what type of vehicle is shown?
- Turkish receipt: What is the total, and what is the cashier's name?
- Food: What color is the pepper to the viewer's left of the kebab, and how many peppers are on the central plate?
An earlier image-captioning check is excluded from these results and is not part of this notebook's question-answer loop.
Notebook
[PaliGemma2-Experiments.ipynb](PaliGemma2-Experiments.ipynb)
The notebook loads the model, accepts the receipt and food images, runs the paired questions, and exports the answers and environment information.
How to Run
1. Open the notebook in Google Colab and select a GPU runtime.
2. Obtain access to the model on Hugging Face if prompted by its model page.
3. Run the installation cell. It installs transformers, accelerate, huggingface_hub, pillow, and pandas. The notebook uses the PyTorch installation provided by Colab.
4. Restart the Colab session after installation, then continue from the next cell.
5. Sign in through notebook_login() and run the model-loading cell.
6. The car image downloads automatically. Upload the receipt and food images separately when their upload cells run. Use the same images as the original experiment to compare with the recorded answers.
7. Run the question-answer loop and the final export cell.
The recorded session reported a Tesla T4 with 14.6 GiB of GPU memory. The notebook loads the model in bfloat16 with device_map="auto". The 10B model's weights require roughly 20 GB at this precision, before additional runtime memory, so it does not fit entirely in that GPU's memory. Automatic placement may use CPU memory and make inference slow. Inspect the printed device map; the notebook also records it for new runs.
Generation Settings
- Model: google/paligemma2-10b-mix-448
- Input resolution: 448 × 448 pixels
- Prompt format: <image>answer {language} {question}
- Language codes: en and tr
- Sampling: disabled with do_sample=False
- Maximum new tokens: 30 per answer
- Additional fine-tuning: none
Recorded Answers
These answers come from my original Colab execution log. They are also included in the notebook's final Markdown cell. The code cells are saved without execution outputs; running them produces fresh results.
Car
- Color — English: blue; Turkish: aqua.
- Vehicle — English: car; Turkish: Resimde bir Volkswagen Beetle gösterilmektedir.
Receipt
- Total — English: 90.20; Turkish: 90,20.
- Cashier — English: ahmet; Turkish: Ahmet.
Food
- Left pepper color — English: green; Turkish: Yeşil.
- Number of peppers — English: 2; Turkish: Iki.
The receipt and food answers matched the reference answers in both languages. I treated decimal separators and capitalization differences as equivalent. The car color answer aqua was relevant to the image but did not follow the requested Turkish language. The extra Volkswagen Beetle identification was not evaluated separately from the vehicle category.
Exported Files
- experiment_results.csv: image names, questions, languages, reference answers, generated answers, and empty columns for manual content and language assessments. The notebook saves it after each response.
- experiment_metadata.json: model identifier and available commit information, Python and package versions, GPU name, device placement, generation settings, and original image dimensions.
The final cell downloads both files. Content accuracy and language adherence are assessed manually; the notebook does not calculate an automatic accuracy score or measure response time.
Limitations
This is an exploratory test with three selected images. Twelve answers do not represent twelve independent images, and the results cannot establish general English or Turkish performance. Answers containing only numbers or names also say little about language fluency.
Only one model size and resolution were tested, so this experiment does not test the paper's scaling findings. The original run did not record exact package versions or CPU/GPU layer placement. The notebook records these details for future runs, but its installation cell does not pin dependency versions.
A follow-up could compare 224 × 224 and 448 × 448 checkpoints of the same model size on more Turkish receipts and signs, using the same questions and hardware.
References
- PaliGemma 2 paper
- PaliGemma 2 10B Mix 448 model card
- Car image from Hugging Face documentation
The receipt and food images were uploaded manually. Their original source links have not yet been documented.

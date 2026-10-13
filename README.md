# PaliGemma 2: English and Turkish Questions and a Resolution Comparison

A small experiment prepared for my Freya Research Fellows article on PaliGemma 2. It includes paired English and Turkish questions using the 10B–448 Mix checkpoint, followed by a comparison of the 3B–224 and 3B–448 Mix checkpoints on one Turkish receipt question. No additional fine-tuning was performed.

## Notebook

[PaliGemma2_Freya_Deney.ipynb](https://github.com/abdullahcankibritoglu/paligemma2-experiment/blob/main/PaliGemma2_Freya_Deney.ipynb)

The notebook contains the experiment code and saved Colab outputs. It prints answers directly; it does not export CSV results or a metadata JSON file. An initial car-captioning check is separate from the question-answer results below.

## Experiment 1: English and Turkish Questions

I used `google/paligemma2-10b-mix-448` with three images: a car, a Turkish receipt, and a plate of food. Two questions per image, each asked in both languages, produced 12 recorded answers.

| Question | English output | Turkish output |
| --- | --- | --- |
| Car color | blue | aqua |
| Vehicle type | car | Resimde bir Volkswagen Beetle gösterilmektedir. |
| Receipt total | 90.20 | 90,20 |
| Cashier's name | ahmet | Ahmet |
| Pepper color, to the viewer's left of the kebab | green | Yeşil. |
| Number of peppers on the central plate | 2 | Iki. |

I treated decimal separators and capitalization differences as equivalent. The receipt and food answers matched the reference answers in both languages. The car color answer “aqua” was relevant to the image but did not follow the requested Turkish language. I did not separately evaluate the Volkswagen Beetle identification beyond the vehicle category.

## Experiment 2: Does Higher Resolution Help on the Same Receipt?

I compared two checkpoints of the same nominal model size on the same receipt and question:

- Question: `Fişteki genel toplam tutarı nedir?`
- Expected answer: `90,20`
- Source image dimensions: 293 × 530 pixels.

| Checkpoint | Input resolution | Recorded answer |
| --- | --- | --- |
| `google/paligemma2-3b-mix-224` | 224 × 224 | 90,20 |
| `google/paligemma2-3b-mix-448` | 448 × 448 | 90,20 |

For this receipt and question, 224 × 224 was sufficient: increasing the input resolution did not improve answer accuracy. The total is relatively large text, so this example does not establish how the checkpoints compare on smaller text or harder receipts. The two checkpoints also differ in training; this is not a controlled test of resolution alone. These two additional answers bring the question-answer total to 14 across three source images.

## Settings and Hardware

| Setting | English/Turkish experiment | Resolution comparison |
| --- | --- | --- |
| Checkpoint | 10B Mix 448 | 3B Mix 224 and 3B Mix 448 |
| Weight dtype in code | `torch.bfloat16` | `torch.float16` |
| Device placement | `device_map="auto"` | `device_map={"": 0}` |
| Sampling | `do_sample=False` | `do_sample=False` |
| Maximum new tokens per answer | 30 | 30 |
| Prompt | `answer {language} {question}` | `<image>answer tr {question}` |

The car question cells explicitly include `<image>`; the original receipt and food question cells omit it and rely on processor inference of the image token. The initial captioning check uses `describe en` and a 100-token limit and is excluded from the 14 answers.

The recorded Colab session used a Tesla T4 with approximately 14.6 GiB of GPU memory. The 10B model's bfloat16 weights alone require roughly 20 GB, so they do not fit entirely on this GPU. Automatic device placement can use CPU memory; the original execution did not record the layer allocation. Neither experiment measured response time or inference cost.

## How to Run

1. Open the linked notebook in Google Colab and select a GPU runtime.
2. Run the dependency installation cells and obtain Hugging Face model access if prompted. Sign in with `notebook_login()`.
3. Load the 10B–448 checkpoint and run the car question cells. The car image downloads automatically.
4. Upload the receipt when prompted, then run the total and cashier questions. Upload the food image at the next upload cell and run the pepper questions.
5. In the follow-up section, upload the receipt again. The food section reuses the variable name `receipt_image`, so this upload is necessary to restore the receipt input.
6. Run the 3B–224 model-loading and question cells, followed by the 3B–448 cells. The notebook deletes the previous model between these checkpoints and prints both answers for comparison.

The notebook retains the package-installation and Pillow troubleshooting cells from the original Colab session. If a package update requires a runtime restart, rerun imports and reload the necessary images and model. Package versions are not pinned, so a new run may differ from the recorded environment.

## Limitations and Next Step

This is an exploratory experiment with three selected images, not a benchmark. Fourteen answers are not fourteen independent images. Number-only and name-only answers provide little evidence about language fluency. The English/Turkish comparison changes both the question language and the prompt language code, so their effects cannot be separated.

The resolution comparison covers only one question about large text on a small source image. A useful next test would use sharper, higher-resolution receipt photographs and ask about a small product price or the time, keeping questions and generation settings fixed. Accuracy and response time should be measured separately. That harder-receipt test has not yet been performed here.

## References and Image Sources

- [PaliGemma 2 paper](https://arxiv.org/abs/2412.03555)
- [10B Mix 448 model card](https://huggingface.co/google/paligemma2-10b-mix-448)
- [3B Mix 224 model card](https://huggingface.co/google/paligemma2-3b-mix-224)
- [3B Mix 448 model card](https://huggingface.co/google/paligemma2-3b-mix-448)
- [Car image: Hugging Face documentation](https://huggingface.co/datasets/huggingface/documentation-images/blob/main/transformers/tasks/car.jpg)
- [Receipt image: Ekonomik Teknoloji](https://ekonomikteknoloji.com/images/thumb/fisler/market/market-fis.jpg)
- [Food image: Google-hosted thumbnail used in the article](https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTdsePiQif6tHV9cIQA5c1mkKx6HiACVieXkyqqf6fffw&s=10)

The receipt and food images are uploaded manually in Colab. The food link is the thumbnail used in the experiment, not an independently verified original publisher attribution.

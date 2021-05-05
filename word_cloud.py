from wordcloud import WordCloud, ImageColorGenerator, STOPWORDS
import matplotlib.pyplot as plt
from PIL import Image
import pandas as pd
import numpy as np
from text_processing import word_cut, word_clean, transform_float


def generate_word_cloud(text, image_path=None):
    image_colors = 'blue'
    mask_logo = None

    if image_path:
        mask_logo = np.array(Image.open(image_path))
        image_colors = ImageColorGenerator(mask_logo)

    font_path = "C:\\Windows\\Fonts\\simfang.ttf"
    wc = WordCloud(font_path=font_path, background_color="white", max_words=2000, max_font_size=90, random_state=1, mask=mask_logo)
    wc.generate_from_text(text)

    plt.figure(figsize=[10, 10])
    if image_path:
        plt.imshow(wc.recolor(color_func=image_colors), interpolation="bilinear")
    plt.axis('off')
    plt.show()


df = pd.read_csv('concat_text.csv', encoding='utf-8')

df['text'] = df['text'].apply(transform_float)
df['text'] = df['text'].apply(word_clean)

texts = df['text'].apply(word_cut)
words = []
for text in texts:
    for word in text:
        words.append(word)

generate_word_cloud(str(words),image_path='1.jpg')

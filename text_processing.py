import jieba
import re


# 处理文本所用的各种函数
def word_cut(text):
    return [word for word in jieba.cut(text) if word.strip()]


def word_clean(raw_text):
    stopwords = [line.rstrip() for line in open('stop_words_zh.txt', 'r', encoding='utf-8')]
    # 1. 使用正则表达式去除非中文字符
    filter_pattern = re.compile('[^\u4E00-\u9FD5]+')
    chinese_only = filter_pattern.sub('', raw_text)

    # 2. 使用jieba进行分词,精确模式
    words_lst = jieba.cut(chinese_only, cut_all=False)

    # 3. 去除停用词
    clean_words = []
    for word in words_lst:
        if word not in stopwords:
            clean_words.append(word)
    return ''.join(clean_words)


def transform_float(raw_text):
    text = str(raw_text)
    return text

def rumor_clean(text):
    text = str(text)
    pattern = re.compile('http.*?[\u4e00-\u9fa5]')
    text = pattern.sub('',text)
    return text
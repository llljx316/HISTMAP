from wordcloud import WordCloud
import matplotlib.pyplot as plt
import re

# 文件路径
file_path = 'word_frequency_log.txt'  # 替换为你的文件路径

# 读取文件并解析单词频率
def read_word_frequencies(file_path):
    common_words_ls=["image","shows","a","large,","with","and","very","on","top","shaped","object","large","base","small","map","of","the","shown","in","branch","hole","center","goat","cylindrical","base","field","foreground","background","it", "is"]
    word_freq = {}
    with open(file_path, 'r') as file:
        for line in file:
            word, freq = line.split(': ')
            # 标准化单词（去除标点符号，转换为小写）
            clean_word = re.sub(r'[^\w\s]', '', word).lower().strip()
            freq = int(freq.strip())
            if clean_word in common_words_ls:
                continue
            
            # 合并单词频率
            if clean_word in word_freq:
                word_freq[clean_word] += freq
            else:
                word_freq[clean_word] = freq
    return word_freq

# 生成词云
def generate_wordcloud(word_freq):
    output_image_path = 'wordcloud.pdf'  # 保存为 PDF 文件
    wordcloud = WordCloud(
        width=800,
        height=400,
        background_color='white'
    ).generate_from_frequencies(word_freq)
    
    plt.figure(figsize=(12, 6))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.savefig(output_image_path, format='pdf', bbox_inches='tight')
    plt.savefig("wordcloud.png", format='png', bbox_inches='tight')
    plt.show()

if __name__ == '__main__':
    word_freq = read_word_frequencies(file_path)
    if word_freq:
        generate_wordcloud(word_freq)
    else:
        print("未找到有效的单词频率数据。")

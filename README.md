# Transformer for Machine Translation

## How to use

Train time
```
python3 machine_translation.py --mode train --epochs 25 --min_freq 2 --N 1 --max_len 128 --verbose --plot
```
Inference time
```
python3 machine_translation.py --mode inference --model_load_path weights checkpoint_Transformer_25.pth --model_config_path configs_Transformer_N1.json
```

Output example

```
--- Your Translator ---

Type your sentence in English: I have to go to sleep.

Source sentence: I have to go to sleep.
Translation sentence: Preciso dormir .
```

## Dataset

The dataset used has sentences translated from English to Portuguese and is provided by [Tatoeba](https://tatoeba.org/). There are several other languages ​​on the website, just download the .tsv file and apply it to the model.

## Short Notes

Short Notes contain mathematical insights and basics concepts related to the components of variants transformer architectures.

- [Foundations of Transformers](https://github.com/paulosantosneto/transformer-variants/blob/main/notes/Foundations.md)
- [PostLN, PreLN and ResiDual Transformer](https://github.com/paulosantosneto/transformer-variants/blob/main/notes/AboutNormalization.md)
- [Multi-query and Multi-group](https://github.com/paulosantosneto/transformer-variants/blob/main/notes/AboutMultihead.md)

## References

### Foundations

[1] Vaswani, Ashish & Shazeer, Noam & Parmar, Niki & Uszkoreit, Jakob & Jones, Llion & Gomez, Aidan & Kaiser, Lukasz & Polosukhin, Illia. (2017). Attention Is All You Need. 

[2] Müller, Rafael & Kornblith, Simon & Hinton, Geoffrey. (2019). When Does Label Smoothing Help?. 

[3] Sennrich, Rico & Haddow, Barry & Birch, Alexandra. (2015). Neural Machine Translation of Rare Words with Subword Units. 

## Optimization

[4] Howard, Jeremy & Ruder, Sebastian. (2018). Universal Language Model Fine-tuning for Text Classification. 328-339. 10.18653/v1/P18-1031. 

[5] Zhang, Tianyi & Wu, Felix & Katiyar, Arzoo & Weinberger, Kilian & Artzi, Yoav. (2020). Revisiting Few-sample BERT Fine-tuning. 

### About Normalization

[6] Xiong, Ruibin & Yang, Yunchang & He, Di & Zheng, Kai & Shuxin, Zheng & Xing, Chen & Zhang, Huishuai & Lan, Yanyan & Wang, Liwei & Liu, Tie-Yan. (2020). On Layer Normalization in the Transformer Architecture. 

[7] Liu, Liyuan & Liu, Xiaodong & Gao, Jianfeng & Chen, Weizhu & Han, Jiawei. (2020). Understanding the Difficulty of Training Transformers. 5747-5763. 10.18653/v1/2020.emnlp-main.463. 

### About Multi-head Attention

[8] Shazeer, Noam. (2019). Fast Transformer Decoding: One Write-Head is All You Need. 

[9] Ainslie, Joshua & Lee-Thorp, James & Jong, Michiel & Zemlyanskiy, Yury & Lebrón, Federico & Sanghai, Sumit. (2023). GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints. 

## Evaluation

[10] Papineni, Kishore & Roukos, Salim & Ward, Todd & Zhu, Wei Jing. (2002). BLEU: a Method for Automatic Evaluation of Machine Translation. 10.3115/1073083.1073135. 


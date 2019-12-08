# 情感细粒度分类说明
1. BERT
  - 有待进一步测试，只为记录学习结果。如果只是二分类的话，应该都没啥问题。
  - 两个classifier均可跑通，预训练的结果如下:
    > global_step = 100  
      loss = 1.7699858  
      masked_lm_accuracy = 0.71597946  
      masked_lm_loss = 1.3563018  
      next_sentence_accuracy = 0.8275  
      next_sentence_loss = 0.41689202
  - 用于训练的model，采用融合label放进去跑，结果如下(前两个指标有点问题，之后查看)：
    > eval_accuracy = 0.05  
      eval_loss = 0.2660327  
      global_step = 39375  
      loss = 0.26603246  
  - 采用标签列表放进去跑（未编码），结果如下（应该需要编码，需要进一步研究）：
    > 0 = 0.5  
      1 = 0.5  
      10 = 0.5  
      11 = 0.5  
      12 = 0.5  
      13 = 0.5  
      14 = 0.5  
      15 = 0.5  
      16 = 0.5  
      17 = 0.5  
      18 = 0.5  
      19 = 0.5  
      2 = 0.5  
      3 = 0.5  
      4 = 0.5  
      5 = 0.5  
      6 = 0.5  
      7 = 0.5  
      8 = 0.5  
      9 = 0.5  
      eval_loss = -327.7296  
      global_step = 45000  
      loss = -327.73026  
2. Keras+LSTM
  - 采用Multi_Outputs，用了20层的模型，准确率具体看Notebook文件。

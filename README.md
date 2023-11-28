# LLAMA 醫療對話繁體中文
在當今數位時代，我們習慣於迅速獲得資訊。但在特定專業領域，如醫學，深厚的專業知識才是關鍵，這常使得大眾在面臨相關問題時難以獲得即時且正確的指導。當前的大型語言模型如GPT或LLAMA雖能回答普通問題，但尚不足以提供專業知識。為此，我們打算利用開源的大型語言模型開發一個專門針對醫療診斷輔助的模型。該模型旨在根據輸入的症狀迅速且準確地提供可能的診斷，從而成為一個實時且可信賴的醫療諮詢工具，協助人們在健康問題上做出明智的決策。
資料集使用[ChatDoctor](https://github.com/Kent0n-Li/ChatDoctor/tree/main)提供的時十筆醫療對話，並進行翻譯
## 翻譯資料集
|        Class         | Description                          
| :------------------: | :----------------------------------- 
|     `HealthCareMagic`          | [google driver](https://drive.google.com/file/d/1CY1yugiK7anSTQtYF_UmQvgx-rVqr_x5/view?usp=drive_link)

而本專案使用由
FlagAlpha 提供的 [Llama2-Chinese-7b-Chat](https://drive.google.com/file/d/17171E3S6HRH9tFwtnhItrS-Vhms7Ce9O/view?usp=drive_link)  進行 Full-tuning 與 Lora 的訓練

我們也提供訓練完的 Checkpoint 可以進行進一步的研究
## Checkpoint
|        Class         | Description                          
| :------------------: | :----------------------------------- 
|     `Full-tuning`          | [vickt/LLama-chinese-med-chat](https://huggingface.co/vickt/LLama-chinese-med-chat)
|     `Lora`          | [vickt/LLama-chinese-med-chat-lora](https://huggingface.co/vickt/LLama-chinese-med-chat-lora)
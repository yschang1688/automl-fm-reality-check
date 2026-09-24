# Colab（T4）執行筆記

1. **教學筆記本寫死了一把臨時 token**（KDD 會後撤銷）。第 3、7 本用「編輯 → 尋找並取代」把那串 token 換成
   `__import__("google.colab.userdata", fromlist=["get"]).get("TABPFN_TOKEN")`，改讀自己的 Colab secret；第 6 本原本就會讀 secret。
   token 要自己到 Prior Labs 申請，並在授權頁逐版接受 TabPFN 的**非商用**授權，否則權重下載會被擋。
2. **每本筆記本第一次讀 secret 都要授權一次**。授權對話框放著太久會逾時，第 6 本會退回要求手動貼 token 的輸入框——中斷那一格、授權後重跑即可，不必手貼。
3. **Chronos-2 LoRA 微調在 Colab 會 ImportError**：預裝的 torchao 0.10 太舊（peft 要求 >0.16）。先跑 `!uv pip install -q --system "torchao>=0.16"`。
4. **免費帳號同時只能掛一個 GPU 執行階段**。依序跑，每本結束後「執行階段 → 中斷連線並刪除執行階段」騰位。
5. 第 6 本的 thinking mode 在 Prior Labs 雲端 API 上跑（約 190 秒），資料會送出去；教學用的是公開資料集。

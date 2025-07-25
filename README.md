# 🧠 NLP Project - Topic Modeling dengan LDA

Proyek ini merupakan tugas mata kuliah **Kecerdasan Buatan** yang mengimplementasikan **Latent Dirichlet Allocation (LDA)** untuk melakukan **Topic Modeling** pada dataset teks.

----------------------------------------------------------
## ✅ Tujuan
- Mengelompokkan dokumen ke dalam topik menggunakan algoritma LDA.
- Menemukan kata-kata yang paling representatif untuk setiap topik.
- Mengevaluasi hasil distribusi topik terhadap dokumen.

----------------------------------------------------------
## 📌 Fitur
- Menggunakan **dataset Reuters** dari pustaka LDA.
- Membuat model LDA untuk mendeteksi 20 topik.
- Menampilkan kata-kata teratas untuk setiap topik.
- Menampilkan topik dominan untuk setiap dokumen.

----------------------------------------------------------
## 🛠 Teknologi yang Digunakan
- Python 3.x
- NumPy
- LDA (Latent Dirichlet Allocation)
- lda.datasets (built-in dataset Reuters)

----------------------------------------------------------
## ⚙️ Cara Menjalankan di Google Colab

1. Install library:
pip install lda

2. Import library dan dataset:
import numpy as np
import lda
import lda.datasets

3. Load dataset Reuters:
X = lda.datasets.load_reuters()
vocab = lda.datasets.load_reuters_vocab()
titles = lda.datasets.load_reuters_titles()

4. Cek dimensi dataset:
X.shape        # Output: (395, 4258)
X.sum()        # Output: 84010

5. Buat dan latih model LDA:
model = lda.LDA(n_topics=20, n_iter=1500, random_state=1)
model.fit(X)

6. Tampilkan kata-kata top untuk setiap topik:
topic_word = model.topic_word_
n_top_words = 8
for i, topic_dist in enumerate(topic_word):
    topic_words = np.array(vocab)[np.argsort(topic_dist)][:-(n_top_words+1):-1]
    print('Topic {}: {}'.format(i, ' '.join(topic_words)))

Contoh output:
Topic 0: british churchill sale million major letters west britain
Topic 1: church government political country state people party against
Topic 2: elvis king fans presley life concert young death
...

7. Cek topik dominan dari 3 dokumen pertama:
doc_topic = model.doc_topic_
for i in range(3):
    print("{} (top topic: {})".format(titles[i], doc_topic[i].argmax()))

Contoh output:
0 UK: Prince Charles spearheads British royal revolution. LONDON 1996-08-20 (top topic: 8)
1 GERMANY: Historic Dresden church rising from WW2 ashes. DRESDEN, Germany 1996-08-21 (top topic: 13)
2 INDIA: Mother Teresa's condition said still unstable. CALCUTTA 1996-08-23 (top topic: 14)

----------------------------------------------------------
## 📊 Hasil Analisis
- Dataset: Reuters (395 dokumen, 4258 kata)
- Total kata: 84.010
- Jumlah topik: 20
- Contoh topik:
  - Topic 2: elvis king fans presley life concert young death
  - Topic 14: mother teresa heart calcutta charity nun hospital missionaries

----------------------------------------------------------
## 🧾 Lisensi
Proyek ini dibuat untuk keperluan akademik (Tugas Kecerdasan Buatan) dan bebas digunakan untuk pembelajaran.
